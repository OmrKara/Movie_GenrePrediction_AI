"""Train the selected weekly Letterboxd models using the saved 70/15/15 split.

Read movies.csv and genres.csv from a raw-data folder and the fixed ID split
from the split-manifest CSV. The test partition is not used for fitting,
threshold selection, or evaluation here.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import sklearn
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset

from letterboxd_project import LABELS, SEED
from letterboxd_project.models import ProjectMLP

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def load_project_data(data_dir: Path, split_path: Path):
    movies = pd.read_csv(data_dir / "movies.csv", usecols=["id", "description"])
    genres = pd.read_csv(data_dir / "genres.csv", usecols=["id", "genre"])
    manifest = pd.read_csv(split_path)
    genres["genre"] = genres["genre"].astype("string").str.strip()
    grouped = genres.dropna(subset=["id", "genre"]).groupby("id")["genre"].agg(lambda x: sorted(set(x)))
    data = movies.drop_duplicates("id").merge(grouped.rename("genres"), left_on="id", right_index=True, how="inner")
    data = data.dropna(subset=["description"]).copy()
    data["description"] = data["description"].astype(str).str.strip()
    data = data[data.description.ne("")].drop_duplicates("description")
    data = data[data.genres.apply(lambda x: bool(set(x).intersection(LABELS)))].copy()
    data = data.sample(n=10000, random_state=SEED).reset_index(drop=True)
    y = np.column_stack([data.genres.apply(lambda gs: int(label in gs)).to_numpy() for label in LABELS]).astype(np.int64)
    order = pd.Series(np.arange(len(data)), index=data.id).reindex(manifest.id).to_numpy()
    if len(manifest) != 10000 or np.isnan(order).any():
        raise ValueError("Data does not match the saved 10,000-film split manifest.")
    data = data.iloc[order.astype(int)].reset_index(drop=True)
    y = y[order.astype(int)]
    split = manifest.split.to_numpy()
    train_i = np.flatnonzero(split == "train")
    val_i = np.flatnonzero(split == "validation")
    test_i = np.flatnonzero(split == "test")
    if (len(train_i), len(val_i), len(test_i)) != (7020, 1490, 1490):
        raise ValueError("Expected split sizes 7020/1490/1490.")
    return data, y, train_i, val_i


def tune_threshold(y_true, probabilities):
    thresholds = np.arange(0.05, 0.951, 0.01)
    scores = [f1_score(y_true, probabilities >= threshold, average="macro", zero_division=0)
              for threshold in thresholds]
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def calculate_metrics(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_roc_auc": float(roc_auc_score(y_true, probabilities, average="macro")),
        "exact_match_accuracy": float(accuracy_score(y_true, predicted)),
    }


def train_mlp(X_train, y_train, X_val, y_val, hidden, dropout, lr, output_path, device,
              max_epochs=30, patience=4):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = ProjectMLP(X_train.shape[1], hidden=hidden, dropout=dropout, output_dim=y_train.shape[1]).to(device)
    positives = y_train.sum(axis=0)
    pos_weight = torch.tensor((len(y_train) - positives) / positives, dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.float32)))
    generator = torch.Generator().manual_seed(SEED)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, generator=generator)
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)
    best_loss, best_epoch, wait = float("inf"), 0, 0
    best_state, best_prob = None, None
    start = time.time()
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_total, n, batches = 0.0, 0, []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_total += loss_fn(logits, yb).item() * len(yb)
                n += len(yb)
                batches.append(torch.sigmoid(logits).cpu().numpy())
        val_loss = val_total / n
        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
            best_prob = np.concatenate(batches, axis=0)
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    threshold, _ = tune_threshold(y_val, best_prob)
    summary = calculate_metrics(y_val, best_prob, threshold)
    torch.save({"state_dict": model.cpu().state_dict(), "input_dim": int(X_train.shape[1]),
                "hidden": list(hidden), "dropout": float(dropout), "n_labels": int(y_train.shape[1]),
                "pos_weight": pos_weight.cpu().tolist(), "threshold": threshold,
                "best_epoch": best_epoch, "max_epochs": max_epochs, "patience": patience,
                "learning_rate": lr, "seed": SEED}, output_path)
    return {"artifact": f"models/{output_path.name}", "uses_pca": False, "threshold": threshold,
            "best_epoch": int(best_epoch), "epochs_run": int(epoch + 1), "validation_loss": float(best_loss),
            "validation_metrics": summary, "parameters": int(sum(p.numel() for p in model.parameters())),
            "time_seconds": round(time.time() - start, 2), "hidden": list(hidden),
            "dropout": dropout, "learning_rate": lr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Folder containing movies.csv and genres.csv")
    parser.add_argument("--split-manifest", type=Path, default=Path("data/splits/letterboxd_split_u5.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--with-u8-search", action="store_true", help="Train and save the U8 four-setting MLP search")
    parser.add_argument("--only", default=None, help="Optionally retrain one registry model key")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir, prep_dir = args.output / "models", args.output / "preprocessing"
    model_dir.mkdir(exist_ok=True)
    prep_dir.mkdir(exist_ok=True)

    data, y, train_i, val_i = load_project_data(args.data_dir, args.split_manifest)
    descriptions = data.description.to_numpy()
    vectorizer = TfidfVectorizer(max_features=1000, min_df=2, ngram_range=(1, 1),
                                 lowercase=True, norm="l2", dtype=np.float32)
    Xtr_sparse = vectorizer.fit_transform(descriptions[train_i])
    Xva_sparse = vectorizer.transform(descriptions[val_i])
    Xtr, Xva = Xtr_sparse.toarray().astype(np.float32), Xva_sparse.toarray().astype(np.float32)
    pca = PCA(n_components=704, svd_solver="full", random_state=SEED)
    Xtr_pca = np.ascontiguousarray(pca.fit_transform(Xtr))
    Xva_pca = np.ascontiguousarray(pca.transform(Xva))
    joblib.dump(vectorizer, prep_dir / "tfidf_vectorizer.joblib", compress=3)
    joblib.dump(pca, prep_dir / "pca_704.joblib", compress=3)
    (prep_dir / "labels.json").write_text(json.dumps(LABELS, ensure_ascii=False, indent=2), encoding="utf-8")
    ytr, yva = y[train_i], y[val_i]

    classifiers = {
        "u2_knn": (KNeighborsClassifier(n_neighbors=201, weights="distance", n_jobs=-1), True),
        "u3_logistic_regression": (LogisticRegression(C=1.4245014245014245, penalty="l2", solver="lbfgs",
                                                        max_iter=3000, tol=1e-8, random_state=SEED), True),
        "u4_decision_tree": (DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=SEED), True),
        "u5_random_forest": (RandomForestClassifier(n_estimators=200, max_features="sqrt", class_weight=None,
                                                      n_jobs=-1, random_state=SEED), True),
        "u5_gradient_boosting": (HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200,
                                                                  early_stopping=False, random_state=SEED), True),
        "u6_linear_svm": (LinearSVC(C=0.1, class_weight="balanced", max_iter=5000, random_state=SEED), False),
    }
    registry_path = args.output / "model_registry.json"
    if args.only and registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"project": "Letterboxd multi-label genre prediction", "seed": SEED, "labels": LABELS,
                    "split_sizes": {"train": len(train_i), "validation": len(val_i), "test": 1490},
                    "features": {"tfidf_max_features": 1000, "min_df": 2, "ngram": "unigram", "pca_components": 704},
                    "threshold_selection": "shared threshold selected on validation macro-F1; test split not used",
                    "models": {}}
    registry["environment"] = {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                                "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    only_is_classical = args.only in classifiers
    if only_is_classical:
        classifiers = {args.only: classifiers[args.only]}
    elif args.only:
        classifiers = {}
    for name, (base_estimator, uses_pca) in classifiers.items():
        print("Training", name, flush=True)
        start = time.time()
        if name == "u2_knn":
            # Multi-output KNN stores the shared neighbour matrix once; each output column
            # is mathematically the same independent distance-weighted binary vote as OVR.
            model = base_estimator.fit(Xtr_pca, ytr)
        else:
            model = OneVsRestClassifier(base_estimator, n_jobs=1).fit(Xtr_pca if uses_pca else Xtr, ytr)
        Xval = Xva_pca if uses_pca else Xva
        if name == "u6_linear_svm":
            probabilities = expit(model.decision_function(Xval))
        else:
            raw_probabilities = model.predict_proba(Xval)
            if isinstance(raw_probabilities, list):
                probabilities = np.column_stack([p[:, int(np.flatnonzero(classes == 1)[0])]
                                                  for p, classes in zip(raw_probabilities, model.classes_)])
            else:
                probabilities = raw_probabilities
        threshold, _ = tune_threshold(yva, probabilities)
        entry = {"artifact": f"models/{name}.joblib", "uses_pca": uses_pca,
                 "threshold": threshold, "validation_metrics": calculate_metrics(yva, probabilities, threshold),
                 "training_seconds": round(time.time() - start, 2)}
        if name == "u2_knn":
            entry["prediction_strategy"] = "multi-output distance-weighted KNN; equivalent per label to OVR and stores shared neighbors once"
        joblib.dump(model, model_dir / f"{name}.joblib", compress=3)
        registry["models"][name] = entry
        print(name, entry["validation_metrics"], "threshold", round(threshold, 2), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [("u7_mlp", (128, 64), 0.2, 0.001)]
    if args.with_u8_search:
        configs.extend([("u8_mlp_dropout0", (128, 64), 0.0, 0.001),
                        ("u8_mlp_dropout02", (128, 64), 0.2, 0.001),
                        ("u8_mlp_large", (256, 128), 0.2, 0.001),
                        ("u8_mlp_lr0005", (128, 64), 0.2, 0.0005)])
    if args.only and only_is_classical:
        configs = []
    elif args.only:
        configs = [config for config in configs if config[0] == args.only]
    candidates = []
    for name, hidden, dropout, lr in configs:
        print("Training", name, "on", device, flush=True)
        entry = train_mlp(Xtr, ytr, Xva, yva, hidden, dropout, lr, model_dir / f"{name}.pt", device)
        registry["models"][name] = entry
        if name.startswith("u8_"):
            candidates.append((entry["validation_metrics"]["macro_f1"], name))
        print(name, entry["validation_metrics"], "threshold", round(entry["threshold"], 2), flush=True)
    if candidates:
        registry["u8_selected_mlp"] = max(candidates)[1]
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved artifacts under", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
