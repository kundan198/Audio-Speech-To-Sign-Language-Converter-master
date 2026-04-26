import json
from pathlib import Path

import numpy as np


FEATURE_SIZE = 126
TARGET_FRAMES = 64


def load_app_vocab(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def label_slug(label):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def landmarks_to_array(landmarks, target_frames=TARGET_FRAMES):
    frames = landmarks or []
    if not frames:
        return np.zeros((target_frames, FEATURE_SIZE), dtype=np.float32)

    sampled = []
    picks = np.linspace(0, len(frames) - 1, target_frames).round().astype(int)
    for idx in picks:
        sampled.append(frame_to_feature(frames[int(idx)]))
    return np.stack(sampled).astype(np.float32)


def frame_to_feature(frame):
    hands = frame or []
    normalized = []
    for hand in hands[:2]:
        pts = np.array([[float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))] for p in hand[:21]], dtype=np.float32)
        if pts.shape[0] < 21:
            pts = np.pad(pts, ((0, 21 - pts.shape[0]), (0, 0)))
        wrist = pts[0].copy()
        pts = pts - wrist
        scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
        if scale > 1e-6:
            pts = pts / scale
        normalized.append(pts.reshape(-1))

    while len(normalized) < 2:
        normalized.append(np.zeros(63, dtype=np.float32))
    return np.concatenate(normalized, axis=0)


def read_sample(path):
    data = json.loads(Path(path).read_text())
    return data["label"], landmarks_to_array(data.get("landmarks") or [])
