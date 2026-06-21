import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_ROOT = Path(r"D:\School\HK6\NT114 - Đồ án chuyên ngành\archive")
OUTPUT_ROOT = Path(r"D:\School\HK6\NT114 - Đồ án chuyên ngành\dataset-v6-fixed")

SCENARIOS = {
    "data_normal": "data_normal",
    "data_deep_anomaly": "data_anomaly",
    "data_deep_anomaly-v2": "data_deep_anomaly",
    "data_final_cascade_anomaly": "data_final_cascade_anomaly",
}

TARGET_SERVICES = sorted([
    "adservice",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "frontend",
    "paymentservice",
    "productcatalogservice",
    "recommendationservice",
    "shippingservice",
])

# Node metrics đang dùng trong DL/RL
NODE_RAW_COLS = [
    "cpu_usage_millicores",
    "memory_usage_bytes",
    "pod_replicas_count",
    "allocated_cpu_quota_millicores",
    "error_rate_ratio",
    "request_per_second",
    "latency_p50_seconds",
    "latency_p95_seconds",
    "latency_p99_seconds",
]

NODE_NORM_COLS = [
    "cpu_usage_millicores_norm",
    "memory_usage_bytes_norm",
    "pod_replicas_count_norm",
    "allocated_cpu_quota_millicores_norm",
    "error_rate_ratio_norm",
    "request_per_second_norm",
    "latency_p50_seconds_norm",
    "latency_p95_seconds_norm",
    "latency_p99_seconds_norm",
]

# Target metrics đang dùng trong DL
TARGET_RAW_COLS = [
    "target_cpu_usage_millicores",
    "target_request_per_second",
    "target_error_rate_ratio",
    "target_latency_p99_seconds",
]

TARGET_NORM_COLS = [
    "target_cpu_usage_millicores_norm",
    "target_request_per_second_norm",
    "target_error_rate_ratio_norm",
    "target_latency_p99_seconds_norm",
]

TARGET_TO_BASE = {
    "target_cpu_usage_millicores": "cpu_usage_millicores",
    "target_request_per_second": "request_per_second",
    "target_error_rate_ratio": "error_rate_ratio",
    "target_latency_p99_seconds": "latency_p99_seconds",
}

# Edge metrics đang dùng trong DL/RL
EDGE_RAW_COLS = [
    "network_latency_seconds",
    "payload_size_bytes",
    "edge_request_rate_rps",
    "edge_error_rate_ratio",
]

EDGE_NORM_COLS = [
    "network_latency_seconds_norm",
    "payload_size_bytes_norm",
    "edge_request_rate_rps_norm",
    "edge_error_rate_ratio_norm",
]

# Hard limits: không dùng percentile
HARD_GLOBAL_MAX = {
    "error_rate_ratio": 1.0,
    "edge_error_rate_ratio": 1.0,
    "pod_replicas_count": 10.0,
}

# Percentile để tránh outlier làm nén dữ liệu
PERCENTILE = 99.5

# Buffer nhẹ, không nên 1.2 quá lớn
BUFFER = 1.10

EPS = 1e-8


# ============================================================
# HELPERS
# ============================================================

def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return pd.read_csv(path)


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def ensure_service_order(df: pd.DataFrame) -> pd.DataFrame:
    if "service" not in df.columns:
        return df

    df = df[df["service"].isin(TARGET_SERVICES)].copy()
    df["service"] = pd.Categorical(df["service"], categories=TARGET_SERVICES, ordered=True)

    sort_cols = []
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")
    sort_cols.append("service")

    return df.sort_values(sort_cols).reset_index(drop=True)


def normalize_col(df: pd.DataFrame, raw_col: str, norm_col: str, global_max: dict):
    if raw_col not in df.columns:
        print(f"  ⚠ Bỏ qua thiếu cột raw: {raw_col}")
        return

    max_val = float(global_max.get(raw_col, 1.0))
    if max_val <= EPS:
        max_val = 1.0

    df[raw_col] = clean_numeric(df[raw_col])
    df[norm_col] = (df[raw_col] / max_val).clip(0.0, 1.0)


def compute_global_max(input_root: Path):
    values = {col: [] for col in NODE_RAW_COLS + EDGE_RAW_COLS}

    for src_dir in SCENARIOS.keys():
        scenario_path = input_root / src_dir
        node_path = scenario_path / "nodes_data.csv"
        edge_path = scenario_path / "edges_data.csv"

        print(f"[*] Quét scenario: {src_dir}")

        node_df = read_csv_safe(node_path)
        edge_df = read_csv_safe(edge_path)

        for col in NODE_RAW_COLS:
            if col in node_df.columns and col not in HARD_GLOBAL_MAX:
                values[col].append(clean_numeric(node_df[col]).to_numpy())

        for col in EDGE_RAW_COLS:
            if col in edge_df.columns and col not in HARD_GLOBAL_MAX:
                values[col].append(clean_numeric(edge_df[col]).to_numpy())

    global_max = {}

    for col in NODE_RAW_COLS + EDGE_RAW_COLS:
        if col in HARD_GLOBAL_MAX:
            global_max[col] = float(HARD_GLOBAL_MAX[col])
            continue

        if len(values[col]) == 0:
            global_max[col] = 1.0
            continue

        arr = np.concatenate(values[col])
        arr = arr[np.isfinite(arr)]

        if len(arr) == 0:
            global_max[col] = 1.0
            continue

        p_val = float(np.percentile(arr, PERCENTILE))
        max_val = float(np.max(arr))

        # Nếu p99.5 quá nhỏ nhưng max có giá trị, dùng max để tránh chia quá bé.
        if p_val <= EPS and max_val > EPS:
            final_val = max_val
        else:
            final_val = p_val * BUFFER

        if final_val <= EPS:
            final_val = 1.0

        global_max[col] = float(final_val)

    return global_max


def convert_nodes(node_df: pd.DataFrame, global_max: dict) -> pd.DataFrame:
    node_df = ensure_service_order(node_df)

    for raw_col, norm_col in zip(NODE_RAW_COLS, NODE_NORM_COLS):
        normalize_col(node_df, raw_col, norm_col, global_max)

    # Nếu có target raw thì normalize lại target theo cùng base metric
    for raw_col, norm_col in zip(TARGET_RAW_COLS, TARGET_NORM_COLS):
        if raw_col not in node_df.columns:
            print(f"  ⚠ Bỏ qua thiếu target raw: {raw_col}")
            continue

        base_col = TARGET_TO_BASE[raw_col]
        max_val = float(global_max.get(base_col, 1.0))

        if max_val <= EPS:
            max_val = 1.0

        node_df[raw_col] = clean_numeric(node_df[raw_col])
        node_df[norm_col] = (node_df[raw_col] / max_val).clip(0.0, 1.0)

    return node_df


def convert_edges(edge_df: pd.DataFrame, global_max: dict) -> pd.DataFrame:
    for raw_col, norm_col in zip(EDGE_RAW_COLS, EDGE_NORM_COLS):
        normalize_col(edge_df, raw_col, norm_col, global_max)

    sort_cols = []
    if "timestamp" in edge_df.columns:
        sort_cols.append("timestamp")
    if "src_service" in edge_df.columns:
        sort_cols.append("src_service")
    if "dst_service" in edge_df.columns:
        sort_cols.append("dst_service")

    if sort_cols:
        edge_df = edge_df.sort_values(sort_cols).reset_index(drop=True)

    return edge_df


def print_global_max(global_max: dict):
    print("\n" + "=" * 70)
    print("GLOBAL_MAX dùng cho train DL/RL và online controller")
    print("=" * 70)
    print("GLOBAL_MAX = {")
    for k, v in global_max.items():
        print(f'    "{k}": {v:.10f},')
    print("}")
    print("=" * 70)


def validate_output(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, scenario_name: str):
    print(f"\n=== Validate {scenario_name} ===")

    norm_cols = NODE_NORM_COLS + TARGET_NORM_COLS
    existing_norm_cols = [c for c in norm_cols if c in nodes_df.columns]

    for col in existing_norm_cols:
        mn = nodes_df[col].min()
        mx = nodes_df[col].max()
        if mn < -1e-6 or mx > 1.000001:
            print(f"  ⚠ {col}: min={mn:.5f}, max={mx:.5f}")
        else:
            print(f"  ✓ {col}: min={mn:.5f}, max={mx:.5f}")

    existing_edge_norm_cols = [c for c in EDGE_NORM_COLS if c in edges_df.columns]

    for col in existing_edge_norm_cols:
        mn = edges_df[col].min()
        mx = edges_df[col].max()
        if mn < -1e-6 or mx > 1.000001:
            print(f"  ⚠ {col}: min={mn:.5f}, max={mx:.5f}")
        else:
            print(f"  ✓ {col}: min={mn:.5f}, max={mx:.5f}")

    if "service" in nodes_df.columns and "timestamp" in nodes_df.columns:
        svc_count = nodes_df.groupby("timestamp")["service"].nunique()
        missing = (svc_count < len(TARGET_SERVICES)).sum()
        if missing > 0:
            print(f"  ⚠ Có {missing} timestamps thiếu service")
        else:
            print("  ✓ Mỗi timestamp đủ 10 services")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("CONVERT DATASET WITH GLOBAL PERCENTILE SCALING")
    print("=" * 70)

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT không tồn tại: {INPUT_ROOT}")

    if OUTPUT_ROOT.exists():
        print(f"[*] Xóa output cũ: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    global_max = compute_global_max(INPUT_ROOT)

    print_global_max(global_max)

    with open(OUTPUT_ROOT / "global_max.json", "w", encoding="utf-8") as f:
        json.dump(global_max, f, indent=4, ensure_ascii=False)

    print(f"\n✓ Đã lưu: {OUTPUT_ROOT / 'global_max.json'}")

    for src_dir, dst_dir in SCENARIOS.items():
        src_path = INPUT_ROOT / src_dir
        dst_path = OUTPUT_ROOT / dst_dir
        dst_path.mkdir(parents=True, exist_ok=True)

        node_path = src_path / "nodes_data.csv"
        edge_path = src_path / "edges_data.csv"

        print("\n" + "-" * 70)
        print(f"[*] Convert {src_dir} -> {dst_dir}")
        print("-" * 70)

        nodes_df = read_csv_safe(node_path)
        edges_df = read_csv_safe(edge_path)

        nodes_out = convert_nodes(nodes_df, global_max)
        edges_out = convert_edges(edges_df, global_max)

        nodes_out.to_csv(dst_path / "nodes_data.csv", index=False)
        edges_out.to_csv(dst_path / "edges_data.csv", index=False)

        validate_output(nodes_out, edges_out, dst_dir)

        print(f"✓ Saved: {dst_path / 'nodes_data.csv'}")
        print(f"✓ Saved: {dst_path / 'edges_data.csv'}")

    print("\n" + "=" * 70)
    print("HOÀN TẤT CHUYỂN ĐỔI DATASET")
    print("=" * 70)
    print(f"Output: {OUTPUT_ROOT}")
    print(f"GLOBAL_MAX: {OUTPUT_ROOT / 'global_max.json'}")


if __name__ == "__main__":
    main()