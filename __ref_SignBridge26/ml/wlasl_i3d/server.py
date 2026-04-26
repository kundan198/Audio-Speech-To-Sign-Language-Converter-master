import argparse
import base64
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.nn as nn

from keypoint_features import FEATURE_SIZE, landmarks_to_array
from pytorch_i3d import InceptionI3d


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models" / "wlasl_i3d" / "pytorch_model.bin"
DEFAULT_LABELS = Path(__file__).with_name("wlasl_class_list.txt")
DEFAULT_APP_VOCAB = Path(__file__).with_name("app_vocab.txt")
DEFAULT_KEYPOINT_MODEL = ROOT / "models" / "keypoint_sign" / "keypoint_model.pt"
DEFAULT_KEYPOINT_LABELS = ROOT / "models" / "keypoint_sign" / "labels.json"


class KeypointGRU(nn.Module):
    def __init__(self, num_classes, hidden_size=192, num_layers=2, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(
            input_size=FEATURE_SIZE,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class KeypointPredictor:
    def __init__(self, model_path=DEFAULT_KEYPOINT_MODEL, labels_path=DEFAULT_KEYPOINT_LABELS, device="cpu"):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.device = torch.device(device)
        self.model = None
        self.labels = []
        self.mtime = None
        self.reload_if_needed()

    def reload_if_needed(self):
        if not self.model_path.exists() or not self.labels_path.exists():
            self.model = None
            self.labels = []
            self.mtime = None
            return False
        mtime = self.model_path.stat().st_mtime
        if self.model is not None and self.mtime == mtime:
            return True
        self.labels = json.loads(self.labels_path.read_text())
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = KeypointGRU(num_classes=len(self.labels))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model
        self.mtime = mtime
        print(f"[keypoint] loaded {len(self.labels)} labels from {self.model_path}")
        return True

    def predict(self, landmarks, top_k=5, min_confidence=0.70):
        if not landmarks or not self.reload_if_needed():
            return None
        tensor = torch.tensor(landmarks_to_array(landmarks), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            probs = torch.softmax(self.model(tensor)[0], dim=0)
            values, indices = torch.topk(probs, k=min(top_k, len(self.labels)))
        predictions = [
            {"label": self.labels[int(index)], "confidence": float(value), "class_id": int(index)}
            for value, index in zip(values.cpu(), indices.cpu())
        ]
        best = predictions[0]
        return {
            "ok": True,
            "text": best["label"] if best["confidence"] >= min_confidence else "[unclear]",
            "predictions": predictions,
            "model": "keypoint-gru",
        }


class WLASLPredictor:
    def __init__(
        self,
        weights_path=DEFAULT_WEIGHTS,
        labels_path=DEFAULT_LABELS,
        app_vocab_path=DEFAULT_APP_VOCAB,
        restrict_to_app_vocab=True,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.labels = self._load_labels(Path(labels_path))
        self.allowed_indices = self._load_allowed_indices(Path(app_vocab_path), restrict_to_app_vocab)

        torch.set_num_threads(max(1, int(os.environ.get("WLASL_TORCH_THREADS", "4"))))
        self.model = InceptionI3d(400, in_channels=3)
        self.model.replace_logits(len(self.labels))
        state = torch.load(Path(weights_path), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _load_labels(self, labels_path):
        labels = []
        for line in labels_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            labels.append(parts[1].strip() if len(parts) > 1 else parts[0].strip())
        return labels

    def _load_allowed_indices(self, app_vocab_path, restrict_to_app_vocab):
        if not restrict_to_app_vocab or not app_vocab_path.exists():
            return None
        vocab = {line.strip().lower() for line in app_vocab_path.read_text().splitlines() if line.strip()}
        allowed = [idx for idx, label in enumerate(self.labels) if label.lower() in vocab]
        return allowed or None

    def predict(self, frame_data_urls, top_k=5, min_confidence=0.55, min_margin=0.15, target_frames=32):
        frames = [self._decode_frame(item) for item in frame_data_urls if isinstance(item, str)]
        frames = [frame for frame in frames if frame is not None]
        if not frames:
            raise ValueError("No decodable frames.")

        tensor = self._frames_to_tensor(frames, target_frames=target_frames).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            scores = torch.mean(logits, dim=2)[0]
            if self.allowed_indices:
                mask = torch.full_like(scores, float("-inf"))
                idx = torch.tensor(self.allowed_indices, device=scores.device)
                mask[idx] = scores[idx]
                scores = mask
            probs = torch.softmax(scores, dim=0)
            values, indices = torch.topk(probs, k=min(top_k, len(self.labels)))

        predictions = [
            {
                "label": self.labels[int(index)],
                "confidence": float(value),
                "class_id": int(index),
            }
            for value, index in zip(values.cpu(), indices.cpu())
        ]
        best = predictions[0]
        runner_up = predictions[1]["confidence"] if len(predictions) > 1 else 0.0
        confident = best["confidence"] >= min_confidence and (best["confidence"] - runner_up) >= min_margin
        text = best["label"] if confident else "[unclear]"
        return {
            "ok": True,
            "text": text,
            "predictions": predictions,
            "model": "wlasl-i3d",
            "restricted_to_app_vocab": bool(self.allowed_indices),
            "min_confidence": min_confidence,
            "min_margin": min_margin,
        }

    def _decode_frame(self, data_url):
        b64 = data_url.split(",", 1)[-1]
        try:
            raw = base64.b64decode(b64)
        except Exception:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _frames_to_tensor(self, frames, target_frames=32):
        indices = np.linspace(0, len(frames) - 1, target_frames).round().astype(int)
        processed = [self._preprocess_frame(frames[i]) for i in indices]
        arr = np.stack(processed).astype(np.float32)
        arr = (arr / 255.0) * 2.0 - 1.0
        arr = np.transpose(arr, (3, 0, 1, 2))
        return torch.from_numpy(arr).unsqueeze(0)

    def _preprocess_frame(self, frame):
        height, width = frame.shape[:2]
        scale = 256.0 / min(height, width)
        resized = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        y = max(0, (resized.shape[0] - 224) // 2)
        x = max(0, (resized.shape[1] - 224) // 2)
        cropped = resized[y:y + 224, x:x + 224]
        if cropped.shape[0] != 224 or cropped.shape[1] != 224:
            cropped = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
        return cropped


def make_handler(predictor, keypoint_predictor, min_confidence, keypoint_min_confidence):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlparse(self.path).path == "/health":
                self._send(200, {"ok": True, "model": "wlasl-i3d"})
                return
            self._send(404, {"ok": False, "error": "Not found."})

        def do_POST(self):
            if urlparse(self.path).path != "/predict":
                self._send(404, {"ok": False, "error": "Not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                keypoint_result = keypoint_predictor.predict(
                    payload.get("landmarks") or [],
                    top_k=int(payload.get("top_k", 5)),
                    min_confidence=float(payload.get("keypoint_min_confidence", keypoint_min_confidence)),
                )
                if keypoint_result and keypoint_result.get("text") != "[unclear]":
                    self._send(200, keypoint_result)
                    return
                result = predictor.predict(
                    payload.get("frames") or [],
                    top_k=int(payload.get("top_k", 5)),
                    min_confidence=float(payload.get("min_confidence", min_confidence)),
                    min_margin=float(payload.get("min_margin", 0.15)),
                    target_frames=int(payload.get("target_frames", 32)),
                )
                self._send(200, result)
            except Exception as exc:
                self._send(500, {"ok": False, "error": str(exc), "model": "wlasl-i3d"})

        def log_message(self, fmt, *args):
            print("[wlasl-i3d] " + fmt % args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve local WLASL I3D predictions.")
    parser.add_argument("--host", default=os.environ.get("WLASL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WLASL_PORT", "8766")))
    parser.add_argument("--weights", default=os.environ.get("WLASL_WEIGHTS", str(DEFAULT_WEIGHTS)))
    parser.add_argument("--labels", default=os.environ.get("WLASL_LABELS", str(DEFAULT_LABELS)))
    parser.add_argument("--app-vocab", default=os.environ.get("WLASL_APP_VOCAB", str(DEFAULT_APP_VOCAB)))
    parser.add_argument("--device", default=os.environ.get("WLASL_DEVICE", "cpu"))
    parser.add_argument("--min-confidence", type=float, default=float(os.environ.get("WLASL_MIN_CONFIDENCE", "0.55")))
    parser.add_argument("--keypoint-model", default=os.environ.get("KEYPOINT_MODEL", str(DEFAULT_KEYPOINT_MODEL)))
    parser.add_argument("--keypoint-labels", default=os.environ.get("KEYPOINT_LABELS", str(DEFAULT_KEYPOINT_LABELS)))
    parser.add_argument("--keypoint-min-confidence", type=float, default=float(os.environ.get("KEYPOINT_MIN_CONFIDENCE", "0.70")))
    parser.add_argument("--full-vocab", action="store_true", help="Use all 2,000 WLASL labels instead of app vocabulary.")
    args = parser.parse_args()

    predictor = WLASLPredictor(
        weights_path=args.weights,
        labels_path=args.labels,
        app_vocab_path=args.app_vocab,
        restrict_to_app_vocab=not args.full_vocab,
        device=args.device,
    )
    keypoint_predictor = KeypointPredictor(
        model_path=args.keypoint_model,
        labels_path=args.keypoint_labels,
        device=args.device,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(
        predictor,
        keypoint_predictor,
        args.min_confidence,
        args.keypoint_min_confidence,
    ))
    print(f"WLASL I3D server listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
