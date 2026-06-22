#!/usr/bin/env python3
"""
train_capacity_model.py
========================
Train Learned Capacity Model từ action_effect_pairs.csv.

Input:
  [r_old_norm, r_new_norm, action, effective_delta,
   cpu_before, rps_before, err_before, lat_before, load_level]

Output:
  [cpu_after, rps_after, err_after, lat_after]

Lưu ý:
- Phù hợp với collect_action_effect.sh mới có effective_delta.
- R_MAX = 10, khớp script thu dữ liệu.
- So sánh learned model với hand-crafted formula trên validation set.
"""

import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

R_MAX = 10.0

# Normalize ranges dựa trên log action-effect hiện tại.
# RPS đã thấy > 500 ở currencyservice load=160, nên không để 200 nữa.
NORM = {
    "cpu":  (0.0, 2000.0),   # millicores
    "rps":  (0.0, 800.0),    # observed có thể > 500
    "err":  (0.0, 1.0),
    "lat":  (0.0, 10.0),     # seconds
    "load": (0.0, 250.0),    # concurrent users
}


FEATURE_NAMES = [
    "r_old_norm",
    "r_new_norm",
    "action",
    "effective_delta",
    "cpu_before_norm",
    "rps_before_norm",
    "err_before_norm",
    "lat_before_norm",
    "load_norm",
]

TARGET_NAMES = [
    "cpu_after_norm",
    "rps_after_norm",
    "err_after_norm",
    "lat_after_norm",
]


# ============================================================
# MODEL
# ============================================================

class CapacityModel(nn.Module):
    """
    Input  (9): r_old_norm, r_new_norm, action, effective_delta,
                cpu_before, rps_before, err_before, lat_before, load_norm
    Output (4): cpu_after, rps_after, err_after, lat_after
    """
    def __init__(self, in_dim=9, hidden=32, out_dim=4):
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
        return torch.sigmoid(self.net(x))


# ============================================================
# DATA
# ============================================================

def norm(x, key):
    lo, hi = NORM[key]
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV thiếu các cột: {missing}")


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)

    required_cols = [
        "load_level",
        "service",
        "r_old",
        "r_new",
        "action",
        "cpu_before",
        "rps_before",
        "err_before",
        "lat_before",
        "cpu_after",
        "rps_after",
        "err_after",
        "lat_after",
    ]
    require_columns(df, required_cols)

    # Nếu CSV cũ chưa có effective_delta thì tự tạo.
    if "effective_delta" not in df.columns:
        df["effective_delta"] = df["r_new"] - df["r_old"]

    numeric_cols = [
        "load_level",
        "r_old",
        "r_new",
        "action",
        "effective_delta",
        "cpu_before",
        "rps_before",
        "err_before",
        "lat_before",
        "cpu_after",
        "rps_after",
        "err_after",
        "lat_after",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_drop = len(df)
    df = df.dropna(subset=numeric_cols).copy()
    dropped = before_drop - len(df)

    if dropped > 0:
        print(f"⚠ Dropped {dropped} invalid rows because of NaN values")

    print(f"Loaded {len(df)} valid action-effect samples")
    print(df.head())

    n_effective = (df["effective_delta"] != 0).sum()
    n_noop = (df["effective_delta"] == 0).sum()

    print(f"\nDataset summary:")
    print(f"  Effective scale events : {n_effective}/{len(df)}")
    print(f"  No-op / hold events    : {n_noop}/{len(df)}")
    print(f"  Services               : {df['service'].nunique()}")
    print(f"  Load levels            : {sorted(df['load_level'].unique().tolist())}")
    print(f"  Max RPS observed       : {df[['rps_before', 'rps_after']].max().max():.2f}")
    print(f"  Max CPU observed       : {df[['cpu_before', 'cpu_after']].max().max():.2f}")
    print(f"  Max Lat observed       : {df[['lat_before', 'lat_after']].max().max():.4f}")
    print(f"  Non-zero err rows      : {((df['err_before'] > 0) | (df['err_after'] > 0)).sum()}")

    X = np.stack([
        df["r_old"].values / R_MAX,
        df["r_new"].values / R_MAX,
        df["action"].values.astype(np.float32),
        df["effective_delta"].values.astype(np.float32),
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


# ============================================================
# BASELINE FORMULA
# ============================================================

def handcrafted_predict(df):
    """
    Baseline hand-crafted cũ để so sánh.
    Đây không phải simulator thật, chỉ dùng làm mốc tham khảo.
    """
    r_new = df["r_new"].values.astype(np.float32)

    cpu_before = norm(df["cpu_before"].values, "cpu")
    rps_before = norm(df["rps_before"].values, "rps")
    err_before = norm(df["err_before"].values, "err")
    lat_before = norm(df["lat_before"].values, "lat")

    # Công thức cũ chủ yếu tác động lên latency/error.
    lat_scale_old = np.sqrt(np.maximum(r_new / 2.0, 1.0))
    err_scale_old = np.power(np.maximum(r_new / 2.0, 1.0), 0.7)

    pred_cpu = cpu_before
    pred_rps = rps_before
    pred_err = np.clip(err_before / err_scale_old, 0.0, 1.0)
    pred_lat = np.clip(lat_before / lat_scale_old, 0.0, 1.0)

    return np.stack([pred_cpu, pred_rps, pred_err, pred_lat], axis=1).astype(np.float32)


# ============================================================
# TRAIN
# ============================================================

def train(csv_path: str, out_path: str, epochs: int = 300, lr: float = 1e-3):
    X, Y, df = load_dataset(csv_path)

    if len(X) < 50:
        print(f"\n⚠ CẢNH BÁO: Chỉ có {len(X)} samples — hơi ít để train ổn định.\n")

    idx = np.arange(len(X))

    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=SEED,
        shuffle=True,
    )

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_val = X[val_idx]
    Y_val = Y[val_idx]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)

    model = CapacityModel(in_dim=X.shape[1], hidden=32, out_dim=Y.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    print(f"\nTraining CapacityModel: {len(X_train)} train / {len(X_val)} val samples")
    print("-" * 70)

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
            mae = torch.abs(val_pred - Y_val_t).mean(dim=0)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch % 30 == 0 or epoch == epochs - 1:
            mae_list = [round(m.item(), 5) for m in mae]
            print(
                f"Epoch {epoch:3d} | "
                f"Train={loss.item():.5f} | "
                f"Val={val_loss:.5f} | "
                f"MAE[cpu,rps,err,lat]={mae_list}"
            )

    if best_state is None:
        raise RuntimeError("Training failed: best_state is None")

    model.load_state_dict(best_state)
    print(f"\n✓ Best val_loss = {best_val:.6f}")

    # ========================================================
    # COMPARE WITH HAND-CRAFTED FORMULA ON VALIDATION SET ONLY
    # ========================================================

    print("\n" + "=" * 70)
    print("SO SÁNH TRÊN VALIDATION SET: Learned Capacity Model vs Hand-crafted formula")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        learned_val = model(X_val_t).numpy()

    df_val = df.iloc[val_idx].copy()
    handcrafted_val = handcrafted_predict(df_val)

    real_val = Y_val

    mae_learned = np.abs(learned_val - real_val).mean(axis=0)
    mae_handcrafted = np.abs(handcrafted_val - real_val).mean(axis=0)

    for i, name in enumerate(["cpu", "rps", "err", "lat"]):
        print(
            f"  {name.upper():>4} MAE — "
            f"hand-crafted: {mae_handcrafted[i]:.5f} | "
            f"learned: {mae_learned[i]:.5f}"
        )

    err_nonzero = ((df["err_before"] > 0) | (df["err_after"] > 0)).sum()

    print("\nNhận xét nhanh:")
    if mae_learned[0] < mae_handcrafted[0]:
        print("  ✓ Learned model tốt hơn baseline ở CPU")
    else:
        print("  ⚠ Learned model chưa tốt hơn baseline ở CPU")

    if mae_learned[1] < mae_handcrafted[1]:
        print("  ✓ Learned model tốt hơn baseline ở RPS")
    else:
        print("  ⚠ Learned model chưa tốt hơn baseline ở RPS")

    if mae_learned[3] < mae_handcrafted[3]:
        print("  ✓ Learned model tốt hơn baseline ở Latency")
    else:
        print("  ⚠ Learned model chưa tốt hơn baseline ở Latency")

    if err_nonzero == 0:
        print("  ⚠ Error Rate toàn bộ dataset bằng 0, chưa thể đánh giá khả năng học error.")
    else:
        if mae_learned[2] < mae_handcrafted[2]:
            print("  ✓ Learned model tốt hơn baseline ở Error")
        else:
            print("  ⚠ Learned model chưa tốt hơn baseline ở Error")

    # ========================================================
    # SAVE
    # ========================================================

    torch.save({
        "model_state": model.state_dict(),
        "in_dim": X.shape[1],
        "hidden": 32,
        "out_dim": Y.shape[1],
        "norm_ranges": NORM,
        "r_max": R_MAX,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "best_val_loss": best_val,
        "n_samples": len(X),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "mae_learned_val": mae_learned,
        "mae_handcrafted_val": mae_handcrafted,
    }, out_path)

    print(f"\n✓ Saved Capacity Model → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./action_effect_data/action_effect_pairs.csv")
    ap.add_argument("--out", default="./capacity_model.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    train(args.csv, args.out, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()