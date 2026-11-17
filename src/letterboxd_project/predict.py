"""Predict five genre labels from film descriptions using saved artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from letterboxd_project import LABELS
from letterboxd_project.models import ProjectMLP

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def predict(descriptions, model_name="u3_logistic_regression", artifact_dir=Path("artifacts")):
    artifact_dir = Path(artifact_dir)
    registry = json.loads((artifact_dir / "model_registry.json").read_text(encoding="utf-8"))
    if model_name == "u8_selected_mlp":
        model_name = registry["u8_selected_mlp"]
    entry = registry["models"][model_name]
    vectorizer = joblib.load(artifact_dir / "preprocessing" / "tfidf_vectorizer.joblib")
    x_sparse = vectorizer.transform(pd.Series(descriptions).fillna("").astype(str))

    if model_name.endswith("mlp") or model_name.startswith("u8_"):
        checkpoint_path = artifact_dir / entry["artifact"]
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = ProjectMLP(checkpoint["input_dim"], tuple(checkpoint["hidden"]),
                           checkpoint["dropout"], checkpoint["n_labels"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        x = torch.from_numpy(x_sparse.toarray().astype(np.float32))
        with torch.no_grad():
            probabilities = torch.sigmoid(model(x)).numpy()
    else:
        model = joblib.load(artifact_dir / entry["artifact"])
        if entry["uses_pca"]:
            pca = joblib.load(artifact_dir / "preprocessing" / "pca_704.joblib")
            x = pca.transform(x_sparse.toarray())
        else:
            x = x_sparse
        if model_name == "u6_linear_svm":
            probabilities = expit(model.decision_function(x))
        else:
            probabilities = model.predict_proba(x)
            if isinstance(probabilities, list):
                probabilities = np.column_stack([
                    p[:, int(np.flatnonzero(classes == 1)[0])]
                    for p, classes in zip(probabilities, model.classes_)
                ])

    threshold = float(entry["threshold"])
    result = {f"{label}_prob": probabilities[:, j] for j, label in enumerate(LABELS)}
    result.update({f"{label}_pred": (probabilities[:, j] >= threshold).astype(int)
                   for j, label in enumerate(LABELS)})
    return pd.DataFrame(result), probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path, help="CSV containing a description column and optionally id")
    parser.add_argument("--model", default="u3_logistic_regression", help="Key from artifacts/model_registry.json")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input_csv)
    if "description" not in frame:
        raise ValueError("Input CSV must have a 'description' column.")
    prediction_frame, _ = predict(frame["description"], args.model, args.artifacts)
    if "id" in frame:
        prediction_frame.insert(0, "id", frame["id"].to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(args.output, index=False)
    print("Saved", args.output.resolve())


if __name__ == "__main__":
    main()
