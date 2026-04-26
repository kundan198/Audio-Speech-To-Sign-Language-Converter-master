import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from keypoint_features import FEATURE_SIZE, read_sample


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "sign_samples"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "keypoint_sign"


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


def load_dataset(data_dir, min_samples):
    paths = sorted(Path(data_dir).glob("*/*.json"))
    rows = [read_sample(path) for path in paths]
    counts = Counter(label for label, _ in rows)
    labels = sorted(label for label, count in counts.items() if count >= min_samples)
    if len(labels) < 2:
        raise SystemExit(f"Need at least two labels with {min_samples}+ samples. Current counts: {dict(counts)}")

    label_to_id = {label: idx for idx, label in enumerate(labels)}
    xs, ys = [], []
    for label, array in rows:
        if label not in label_to_id:
            continue
        xs.append(array)
        ys.append(label_to_id[label])
    return np.stack(xs), np.array(ys, dtype=np.int64), labels, counts


def main():
    parser = argparse.ArgumentParser(description="Train SignBridge's local MediaPipe keypoint classifier.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-samples", type=int, default=8)
    args = parser.parse_args()

    x_np, y_np, labels, counts = load_dataset(args.data_dir, args.min_samples)
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    dataset = TensorDataset(x, y)

    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    model = KeypointGRU(num_classes=len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    best_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = torch.argmax(model(xb), dim=1)
                correct += int((pred == yb).sum())
                total += len(yb)
        val_acc = correct / max(total, 1)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"epoch={epoch:03d} loss={total_loss / max(train_size, 1):.4f} val_acc={val_acc:.3f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "num_classes": len(labels),
        "feature_size": FEATURE_SIZE,
    }, output_dir / "keypoint_model.pt")
    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    (output_dir / "training_summary.json").write_text(json.dumps({
        "labels": labels,
        "sample_counts": {label: counts[label] for label in labels},
        "best_val_accuracy": best_acc,
        "total_samples": int(len(dataset)),
    }, indent=2))
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
