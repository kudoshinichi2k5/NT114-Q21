#!/usr/bin/env python3
"""
train_capacity_model.py
========================
Train Capacity Model THẬT từ action_effect_pairs.csv
(thay cho công thức sqrt/pow hand-crafted trong RL env)

Input  : [r_old, r_new, action, load_level,
          cpu_before, rps_before, err_before, lat_before]
Output : [cpu_after, rps_after, err_after, lat_after]   (delta hoặc absolute)

Model: MLP nhỏ (đủ cho bài toán low-dimensional, tránh overfit
       với dataset nhỏ — thường vài trăm action-effect samples).

Usage:
  python3 train_capacity_model.py \
      --csv ./action_effect_data/action_effect_pairs.csv \
      --out ./capacity_model.pt
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────
#  MODEL: Capacity Model (nhỏ, vì input/output đều low-dim)
# ─────────────────────────────────────────────────────────────

class CapacityModel(nn.Module):
    """
    Input  (8) : r_old_norm, r_new_norm, action,
                 cpu_before, rps_before, err_before, lat_before, load_norm
    Output (4) : cpu_after, rps_after, err_after, lat_after  (normalized [0,1])
    """
    def __init__(self, in_dim=8, hidden=32, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))   # output trong [0,1]


# ─────────────────────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────

# Ranges để normalize — khớp với NORM_RANGES trong autoscaler.py
NORM = {
    "cpu":  (0.0, 2000.0),    # millicores
    "rps":  (0.0, 200.0),
    "err":  (0.0, 1.0),
    "lat":  (0.0, 10.0),      # seconds
    "load": (0.0, 250.0),    # concurrent users
}


def norm(x, key):
    lo, hi = NORM[key]
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} action-effect samples")
    print(df.head())

    # Lọc các trial action=0, r_old==r_new (no-op thật) —
    # vẫn giữ lại vì cũng là ground truth hữu ích (baseline behavior)
    n_action = (df["action"] != 0).sum()
    print(f"  Action != 0 (real scale events): {n_action}/{len(df)}")

    R_MAX = 5.0  # khớp HPA --max=5

    X = np.stack([
        df["r_old"].values / R_MAX,
        df["r_new"].values / R_MAX,
        df["action"].values.astype(np.float32),       # đã ∈ {-1,0,1}
        norm(df["cpu_before"].values, "cpu"),
        norm(df["rps_before"].values, "rps"),
        norm(df["err_before"].values, "err"),
        norm(df["lat_before"].values, "lat"),
        norm(df["load_level"].values, "load"),
    ], axis=1).astype(np.float32)

    Y = np.stack([
        norm(df["cpu_after"].values, "cpu"),
        norm(df["rps_after"].values, "rps"),
        norm(df["err_after"].values, "err"),
        norm(df["lat_after"].values, "lat"),
    ], axis=1).astype(np.float32)

    return X, Y, df


# ─────────────────────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────────────────────

def train(csv_path: str, out_path: str, epochs: int = 300, lr: float = 1e-3):
    X, Y, df = load_dataset(csv_path)

    if len(X) < 30:
        print(f"\n⚠ CẢNH BÁO: Chỉ có {len(X)} samples — quá ít để train ổn định.")
        print("  Khuyến nghị: chạy collect_action_effect.sh thêm với nhiều load_level/trials hơn.")
        print("  Vẫn tiếp tục train với dataset hiện có...\n")

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    Y_val_t   = torch.tensor(Y_val, dtype=torch.float32)

    model = CapacityModel(in_dim=X.shape[1], hidden=32, out_dim=Y.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    print(f"\nTraining CapacityModel: {len(X_train)} train / {len(X_val)} val samples")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, Y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 30 == 0 or epoch == epochs - 1:
            mae = torch.abs(val_pred - Y_val_t).mean(dim=0)
            print(f"Epoch {epoch:3d} | Train={loss.item():.4f} | Val={val_loss:.4f} | "
                  f"MAE[cpu,rps,err,lat]={[round(m.item(),4) for m in mae]}")

    model.load_state_dict(best_state)
    print(f"\n✓ Best val_loss = {best_val:.5f}")

    # ── Sanity check: so sánh với công thức hand-crafted cũ ────
    print("\n" + "=" * 60)
    print("SO SÁNH: Capacity Model học được vs Hand-crafted formula")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        all_pred = model(torch.tensor(X, dtype=torch.float32)).numpy()

    # Công thức cũ trong RL env (để so sánh)
    r_old_real = df["r_old"].values
    r_new_real = df["r_new"].values
    lat_scale_old = np.sqrt(np.maximum(r_new_real / 2.0, 1.0))
    err_scale_old = np.power(np.maximum(r_new_real / 2.0, 1.0), 0.7)

    lat_before_norm = norm(df["lat_before"].values, "lat")
    err_before_norm = norm(df["err_before"].values, "err")

    handcrafted_lat = np.clip(lat_before_norm / lat_scale_old, 0, 1)
    handcrafted_err = np.clip(err_before_norm / err_scale_old, 0, 1)

    real_lat = norm(df["lat_after"].values, "lat")
    real_err = norm(df["err_after"].values, "err")

    mae_handcrafted_lat = np.abs(handcrafted_lat - real_lat).mean()
    mae_learned_lat      = np.abs(all_pred[:, 3] - real_lat).mean()
    mae_handcrafted_err  = np.abs(handcrafted_err - real_err).mean()
    mae_learned_err       = np.abs(all_pred[:, 2] - real_err).mean()

    print(f"  Latency MAE — hand-crafted: {mae_handcrafted_lat:.4f} | learned: {mae_learned_lat:.4f}")
    print(f"  Error   MAE — hand-crafted: {mae_handcrafted_err:.4f} | learned: {mae_learned_err:.4f}")
    if mae_learned_lat < mae_handcrafted_lat and mae_learned_err < mae_handcrafted_err:
        print("  ✓ Capacity Model học được tốt hơn formula cũ")
    else:
        print("  ⚠ Cần thêm data hoặc tune lại — formula cũ có thể vẫn tốt hơn ở 1 số target")

    # ── Save ───────────────────────────────────────────────────
    torch.save({
        "model_state": model.state_dict(),
        "in_dim": X.shape[1],
        "hidden": 32,
        "out_dim": Y.shape[1],
        "norm_ranges": NORM,
        "best_val_loss": best_val,
        "n_samples": len(X),
    }, out_path)
    print(f"\n✓ Saved Capacity Model → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",    default="./action_effect_data/action_effect_pairs.csv")
    ap.add_argument("--out",    default="./capacity_model.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr",     type=float, default=1e-3)
    args = ap.parse_args()

    train(args.csv, args.out, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()