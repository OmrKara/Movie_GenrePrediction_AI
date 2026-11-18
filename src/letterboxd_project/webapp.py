"""Run the local browser interface for the Letterboxd genre models."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from letterboxd_project import LABELS
from letterboxd_project.predict import predict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts"
WEB_DIR = PROJECT_ROOT / "web"
MAX_DESCRIPTION_LENGTH = 5000
MODEL_TITLES = {
    "u2_knn": "U2 · kNN",
    "u3_logistic_regression": "U3 · Lojistik regresyon",
    "u4_decision_tree": "U4 · Karar ağacı",
    "u5_random_forest": "U5 · Rastgele orman",
    "u5_gradient_boosting": "U5 · Gradyan artırma",
    "u6_linear_svm": "U6 · Doğrusal SVM",
    "u7_mlp": "U7 · MLP",
    "u8_mlp_dropout0": "U8 · MLP, dropout 0",
    "u8_mlp_dropout02": "U8 · MLP, dropout 0.2",
    "u8_mlp_large": "U8 · Büyük MLP",
    "u8_mlp_lr0005": "U8 · MLP, öğrenme oranı 0.0005",
    "u8_selected_mlp": "U8 · En iyi doğrulama modeli",
}


def make_handler(artifact_dir: Path):
    registry_path = artifact_dir / "model_registry.json"

    class AppHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, status: int, value: dict) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self._send(status, payload, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if self.path == "/":
                self._serve_file("index.html", "text/html; charset=utf-8")
            elif self.path == "/app.js":
                self._serve_file("app.js", "text/javascript; charset=utf-8")
            elif self.path == "/style.css":
                self._serve_file("style.css", "text/css; charset=utf-8")
            elif self.path == "/api/models":
                try:
                    registry = json.loads(registry_path.read_text(encoding="utf-8"))
                    keys = list(registry["models"])
                    if "u8_selected_mlp" in registry:
                        keys.append("u8_selected_mlp")
                    models = []
                    for key in keys:
                        resolved_key = registry.get("u8_selected_mlp") if key == "u8_selected_mlp" else key
                        metrics = registry["models"][resolved_key].get("validation_metrics", {})
                        models.append({
                            "id": key,
                            "title": MODEL_TITLES.get(key, key),
                            "macro_f1": metrics.get("macro_f1"),
                        })
                    self._send_json(200, {"models": models, "labels": LABELS})
                except (OSError, ValueError, KeyError) as error:
                    self._send_json(500, {"error": f"Model listesi okunamadı: {error}"})
            else:
                self._send_json(404, {"error": "Sayfa bulunamadı."})

        def _serve_file(self, filename: str, content_type: str) -> None:
            try:
                payload = (WEB_DIR / filename).read_bytes()
            except OSError:
                self._send_json(500, {"error": "Web arayüzü dosyası bulunamadı."})
                return
            self._send(200, payload, content_type)

        def do_POST(self) -> None:
            if self.path != "/api/predict":
                self._send_json(404, {"error": "Adres bulunamadı."})
                return
            try:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._send_json(400, {"error": "İstek boyutu okunamadı."})
                    return
                if length <= 0 or length > MAX_DESCRIPTION_LENGTH + 1000:
                    self._send_json(413, {"error": "Açıklama boş olamaz ve en fazla 5.000 karakter olmalıdır."})
                    return
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    self._send_json(400, {"error": "İstek alanları JSON nesnesi olmalıdır."})
                    return
                description = request.get("description", "")
                model_name = request.get("model", "u3_logistic_regression")
                if not isinstance(description, str):
                    self._send_json(400, {"error": "Açıklama metin olmalıdır."})
                    return
                description = description.strip()
                if not description:
                    self._send_json(400, {"error": "Önce bir film açıklaması yaz."})
                    return
                if len(description) > MAX_DESCRIPTION_LENGTH:
                    self._send_json(413, {"error": "Açıklama en fazla 5.000 karakter olabilir."})
                    return
                if not isinstance(model_name, str):
                    self._send_json(400, {"error": "Model seçimi geçersiz."})
                    return

                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                model_keys = set(registry["models"])
                if "u8_selected_mlp" in registry:
                    model_keys.add("u8_selected_mlp")
                if model_name not in model_keys:
                    self._send_json(400, {"error": "Seçilen model bulunamadı."})
                    return

                prediction_frame, probability_array = predict(
                    [description], model_name=model_name, artifact_dir=artifact_dir
                )
                resolved_name = registry.get("u8_selected_mlp") if model_name == "u8_selected_mlp" else model_name
                threshold = float(registry["models"][resolved_name]["threshold"])
                probabilities = {
                    label: float(probability_array[0, index])
                    for index, label in enumerate(LABELS)
                }
                predicted_labels = [
                    label for label in LABELS
                    if int(prediction_frame.iloc[0][f"{label}_pred"]) == 1
                ]
                top_label = max(probabilities, key=probabilities.get)
                self._send_json(200, {
                    "model": model_name,
                    "threshold": threshold,
                    "predicted_labels": predicted_labels,
                    "top_label": top_label,
                    "probabilities": probabilities,
                })
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "İstek JSON biçiminde okunamadı."})
            except (OSError, KeyError, ValueError, TypeError, RuntimeError) as error:
                self._send_json(500, {"error": f"Tahmin oluşturulamadı: {error}"})

        def log_message(self, _format: str, *_args) -> None:
            return

    return AppHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Letterboxd tür tahmini web arayüzünü başlatır.")
    parser.add_argument("--host", default="127.0.0.1", help="Varsayılan olarak yalnızca bu bilgisayarda erişilir.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()

    if not (args.artifacts / "model_registry.json").is_file():
        parser.error(f"Model dosyaları bulunamadı: {args.artifacts}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.artifacts.resolve()))
    print(f"Movie Genre Predictor: http://{args.host}:{args.port}")
    print("Durdurmak için Ctrl+C tuşlarına basın.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSunucu durduruldu.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
