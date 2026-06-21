# ============================================================
# CAPACITY / WORKLOAD PATTERN ANALYSIS FOR ONLINE BOUTIQUE
# Phân tích node + edge metrics để hiệu chỉnh RL capacity_model
# ============================================================

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ============================================================
# CONFIG
# ============================================================

# Máy thật:
# DATA_ROOT = Path(r"D:\School\HK6\NT114 - Đồ án chuyên ngành\dataset-v8")

# Kaggle:
DATA_ROOT = Path("/kaggle/input/datasets/kudo123a/dataset-v8")

OUT_DIR = Path("/kaggle/working/capacity_pattern_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "normal": "data_normal",
    "anomaly_main": "data_anomaly",
    "anomaly_edge": "data_deep_anomaly",
    "anomaly_cascade": "data_final_cascade_anomaly",
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

R_MAX = 10
EPS = 1e-8

# Node metrics bạn đang dùng cho DL/RL
NODE_COLS = {
    "cpu": "cpu_usage_millicores_norm",
    "memory": "memory_usage_bytes_norm",
    "replicas": "pod_replicas_count_norm",
    "alloc_cpu": "allocated_cpu_quota_millicores_norm",
    "error": "error_rate_ratio_norm",
    "rps": "request_per_second_norm",
    "lat_p50": "latency_p50_seconds_norm",
    "lat_p95": "latency_p95_seconds_norm",
    "lat_p99": "latency_p99_seconds_norm",
}

# Edge metrics bạn đang dùng cho GAT
EDGE_COLS = {
    "network_latency": "network_latency_seconds_norm",
    "payload": "payload_size_bytes_norm",
    "edge_rps": "edge_request_rate_rps_norm",
    "edge_error": "edge_error_rate_ratio_norm",
}

# Metrics chính cần xem cho RL capacity model
RL_NODE_METRICS = ["cpu", "error", "lat_p99", "rps"]
RL_EDGE_METRICS = ["network_latency", "edge_rps", "edge_error", "payload"]


# ============================================================
# LOAD DATA
# ============================================================

node_dfs = []
edge_dfs = []

for scenario_name, folder in SCENARIOS.items():
    node_path = DATA_ROOT / folder / "nodes_data.csv"
    edge_path = DATA_ROOT / folder / "edges_data.csv"

    if node_path.exists():
        df = pd.read_csv(node_path)
        df["scenario"] = scenario_name
        if "service" in df.columns:
            df = df[df["service"].isin(TARGET_SERVICES)].copy()
        node_dfs.append(df)
        print(f"✓ Loaded node {scenario_name}: {len(df)} rows")
    else:
        print(f"⚠ Missing node file: {node_path}")

    if edge_path.exists():
        df = pd.read_csv(edge_path)
        df["scenario"] = scenario_name
        edge_dfs.append(df)
        print(f"✓ Loaded edge {scenario_name}: {len(df)} rows")
    else:
        print(f"⚠ Missing edge file: {edge_path}")

if len(node_dfs) == 0:
    raise RuntimeError("Không tìm thấy nodes_data.csv")

nodes = pd.concat(node_dfs, ignore_index=True)

if len(edge_dfs) > 0:
    edges = pd.concat(edge_dfs, ignore_index=True)
else:
    edges = pd.DataFrame()

# Chuẩn hóa numeric
for name, col in NODE_COLS.items():
    if col in nodes.columns:
        nodes[col] = pd.to_numeric(nodes[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

if not edges.empty:
    for name, col in EDGE_COLS.items():
        if col in edges.columns:
            edges[col] = pd.to_numeric(edges[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

nodes["replicas_int"] = np.clip(
    np.rint(nodes[NODE_COLS["replicas"]] * R_MAX),
    1,
    R_MAX,
).astype(int)

print("\nNode rows:", len(nodes))
print("Edge rows:", len(edges))


# ============================================================
# 1. DISTRIBUTION BY SCENARIO
# ============================================================

dist_rows = []

for scenario in sorted(nodes["scenario"].unique()):
    sub = nodes[nodes["scenario"] == scenario]

    for metric_name in RL_NODE_METRICS:
        col = NODE_COLS[metric_name]
        if col not in sub.columns:
            continue

        vals = sub[col].values
        dist_rows.append({
            "type": "node",
            "scenario": scenario,
            "metric": metric_name,
            "p50": np.percentile(vals, 50),
            "p75": np.percentile(vals, 75),
            "p90": np.percentile(vals, 90),
            "p95": np.percentile(vals, 95),
            "p99": np.percentile(vals, 99),
            "max": np.max(vals),
            "mean": np.mean(vals),
            "std": np.std(vals),
            "nonzero_percent": float((vals > 1e-6).mean() * 100),
        })

if not edges.empty:
    for scenario in sorted(edges["scenario"].unique()):
        sub = edges[edges["scenario"] == scenario]

        for metric_name in RL_EDGE_METRICS:
            col = EDGE_COLS[metric_name]
            if col not in sub.columns:
                continue

            vals = sub[col].values
            dist_rows.append({
                "type": "edge",
                "scenario": scenario,
                "metric": metric_name,
                "p50": np.percentile(vals, 50),
                "p75": np.percentile(vals, 75),
                "p90": np.percentile(vals, 90),
                "p95": np.percentile(vals, 95),
                "p99": np.percentile(vals, 99),
                "max": np.max(vals),
                "mean": np.mean(vals),
                "std": np.std(vals),
                "nonzero_percent": float((vals > 1e-6).mean() * 100),
            })

dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(OUT_DIR / "01_metric_distribution_by_scenario.csv", index=False)

print("\n=== Metric distribution summary ===")
print(dist_df.head(20))


# ============================================================
# 2. NODE CORRELATION: RPS vs CPU/LAT/ERR/MEM/ALLOC
# ============================================================

node_corr_rows = []

for scenario in sorted(nodes["scenario"].unique()):
    for svc in TARGET_SERVICES:
        sub = nodes[(nodes["scenario"] == scenario) & (nodes["service"] == svc)].copy()

        if len(sub) < 20:
            continue

        row = {
            "scenario": scenario,
            "service": svc,
            "rows": len(sub),
            "mean_replicas": sub["replicas_int"].mean(),
            "mean_rps": sub[NODE_COLS["rps"]].mean(),
        }

        for metric_name in ["cpu", "memory", "alloc_cpu", "error", "lat_p50", "lat_p95", "lat_p99"]:
            col = NODE_COLS[metric_name]
            if col in sub.columns:
                row[f"corr_rps_{metric_name}"] = sub[NODE_COLS["rps"]].corr(sub[col])
                row[f"corr_replicas_{metric_name}"] = sub["replicas_int"].corr(sub[col])

        node_corr_rows.append(row)

node_corr_df = pd.DataFrame(node_corr_rows)
node_corr_df.to_csv(OUT_DIR / "02_node_correlations_by_service.csv", index=False)

print("\n=== Node correlation summary ===")
print(node_corr_df.describe(numeric_only=True))


# ============================================================
# 3. EDGE CORRELATION: edge_rps vs network_latency/edge_error/payload
# ============================================================

edge_corr_rows = []

if not edges.empty:
    # Xác định tên cột source/destination tự động
    possible_src = ["source", "src", "src_service", "source_service", "from_service"]
    possible_dst = ["destination", "dst", "dst_service", "destination_service", "to_service"]

    src_col = next((c for c in possible_src if c in edges.columns), None)
    dst_col = next((c for c in possible_dst if c in edges.columns), None)

    if src_col is None or dst_col is None:
        print("⚠ Không tìm thấy cột src/dst trong edges. Sẽ phân tích theo scenario tổng.")
        group_cols = ["scenario"]
    else:
        group_cols = ["scenario", src_col, dst_col]

    for keys, sub in edges.groupby(group_cols):
        if len(sub) < 20:
            continue

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {"rows": len(sub)}
        for col_name, key in zip(group_cols, keys):
            row[col_name] = key

        edge_rps_col = EDGE_COLS["edge_rps"]

        for metric_name in ["network_latency", "edge_error", "payload"]:
            col = EDGE_COLS[metric_name]
            if col in sub.columns and edge_rps_col in sub.columns:
                row[f"corr_edge_rps_{metric_name}"] = sub[edge_rps_col].corr(sub[col])

        edge_corr_rows.append(row)

edge_corr_df = pd.DataFrame(edge_corr_rows)
edge_corr_df.to_csv(OUT_DIR / "03_edge_correlations.csv", index=False)

print("\n=== Edge correlation summary ===")
if len(edge_corr_df) > 0:
    print(edge_corr_df.describe(numeric_only=True))
else:
    print("No edge correlation rows.")


# ============================================================
# 4. FIT CAPACITY EXPONENTS FOR NODE METRICS
#
# log(metric) = a*log(rps) + b*log(replicas) + c
# capacity exponent k = -b
# metric ≈ rps^a / replicas^k
# ============================================================

def fit_node_capacity_exponent(df, metric_col, min_rows=30):
    sub = df[[NODE_COLS["rps"], "replicas_int", metric_col]].copy()

    sub = sub[
        (sub[NODE_COLS["rps"]] > 1e-5) &
        (sub[metric_col] > 1e-5) &
        (sub["replicas_int"] >= 1)
    ].copy()

    if len(sub) < min_rows:
        return None

    X = np.stack([
        np.log(sub[NODE_COLS["rps"]].values + EPS),
        np.log(sub["replicas_int"].values + EPS),
    ], axis=1)

    y = np.log(sub[metric_col].values + EPS)

    model = LinearRegression()
    model.fit(X, y)

    coef_rps = float(model.coef_[0])
    coef_rep = float(model.coef_[1])
    k = -coef_rep
    r2 = float(model.score(X, y))

    return {
        "rows": len(sub),
        "rps_exponent": coef_rps,
        "replica_coef_raw": coef_rep,
        "capacity_exponent_k": k,
        "r2_log_model": r2,
        "intercept": float(model.intercept_),
    }


fit_rows = []

for scenario in sorted(nodes["scenario"].unique()):
    for svc in TARGET_SERVICES:
        sub = nodes[(nodes["scenario"] == scenario) & (nodes["service"] == svc)].copy()

        for metric_name in ["cpu", "lat_p99", "error"]:
            col = NODE_COLS[metric_name]
            if col not in sub.columns:
                continue

            res = fit_node_capacity_exponent(sub, col)
            if res is None:
                continue

            fit_rows.append({
                "scenario": scenario,
                "service": svc,
                "metric": metric_name,
                **res,
            })

fit_node_df = pd.DataFrame(fit_rows)
fit_node_df.to_csv(OUT_DIR / "04_node_capacity_exponents.csv", index=False)

print("\n=== Node capacity exponents ===")
if len(fit_node_df) > 0:
    print(fit_node_df.groupby("metric")["capacity_exponent_k"].describe())
else:
    print("No fitted node exponents.")


# ============================================================
# 5. FIT EDGE BOTTLENECK EXPONENTS
#
# Đây không dùng replicas trực tiếp, vì edge không có pod.
# Ta fit relation:
# log(network_latency/error) = a*log(edge_rps) + b*log(payload) + c
# để biết edge_rps/payload có giải thích edge bottleneck không.
# ============================================================

def fit_edge_pressure_model(df, target_col, min_rows=30):
    needed = [EDGE_COLS["edge_rps"], EDGE_COLS["payload"], target_col]
    sub = df[needed].copy()

    sub = sub[
        (sub[EDGE_COLS["edge_rps"]] > 1e-5) &
        (sub[EDGE_COLS["payload"]] > 1e-5) &
        (sub[target_col] > 1e-5)
    ].copy()

    if len(sub) < min_rows:
        return None

    X = np.stack([
        np.log(sub[EDGE_COLS["edge_rps"]].values + EPS),
        np.log(sub[EDGE_COLS["payload"]].values + EPS),
    ], axis=1)

    y = np.log(sub[target_col].values + EPS)

    model = LinearRegression()
    model.fit(X, y)

    return {
        "rows": len(sub),
        "edge_rps_exponent": float(model.coef_[0]),
        "payload_exponent": float(model.coef_[1]),
        "r2_log_model": float(model.score(X, y)),
        "intercept": float(model.intercept_),
    }


edge_fit_rows = []

if not edges.empty:
    possible_src = ["source", "src", "src_service", "source_service", "from_service"]
    possible_dst = ["destination", "dst", "dst_service", "destination_service", "to_service"]

    src_col = next((c for c in possible_src if c in edges.columns), None)
    dst_col = next((c for c in possible_dst if c in edges.columns), None)

    if src_col is not None and dst_col is not None:
        group_cols = ["scenario", src_col, dst_col]
    else:
        group_cols = ["scenario"]

    for keys, sub in edges.groupby(group_cols):
        if len(sub) < 20:
            continue

        if not isinstance(keys, tuple):
            keys = (keys,)

        base_row = {}
        for col_name, key in zip(group_cols, keys):
            base_row[col_name] = key

        for target_name in ["network_latency", "edge_error"]:
            target_col = EDGE_COLS[target_name]
            if target_col not in sub.columns:
                continue

            res = fit_edge_pressure_model(sub, target_col)
            if res is None:
                continue

            edge_fit_rows.append({
                **base_row,
                "target": target_name,
                **res,
            })

edge_fit_df = pd.DataFrame(edge_fit_rows)
edge_fit_df.to_csv(OUT_DIR / "05_edge_pressure_models.csv", index=False)

print("\n=== Edge pressure model summary ===")
if len(edge_fit_df) > 0:
    print(edge_fit_df.groupby("target")[["edge_rps_exponent", "payload_exponent", "r2_log_model"]].describe())
else:
    print("No fitted edge pressure models.")


# ============================================================
# 6. RECOMMENDED CAPACITY MODEL PARAMETERS
# ============================================================

def safe_median(df, metric):
    sub = df[
        (df["metric"] == metric) &
        (df["r2_log_model"] > 0.05) &
        (df["capacity_exponent_k"] > -1.0) &
        (df["capacity_exponent_k"] < 2.0)
    ]
    if len(sub) == 0:
        return None
    return float(sub["capacity_exponent_k"].median())

cpu_alpha = safe_median(fit_node_df, "cpu") if len(fit_node_df) > 0 else None
lat_beta = safe_median(fit_node_df, "lat_p99") if len(fit_node_df) > 0 else None
err_gamma = safe_median(fit_node_df, "error") if len(fit_node_df) > 0 else None

# fallback nếu data không fit được tốt
recommended = {
    "cpu_alpha_from_data": cpu_alpha,
    "lat_beta_from_data": lat_beta,
    "err_gamma_from_data": err_gamma,

    "cpu_alpha_fallback": 0.85,
    "lat_beta_fallback": 0.60,
    "err_gamma_fallback": 0.50,

    "notes": (
        "Use *_from_data only if signs are reasonable and r2_log_model is meaningful. "
        "Otherwise use fallback queueing-inspired exponents."
    )
}

with open(OUT_DIR / "06_recommended_capacity_params.json", "w", encoding="utf-8") as f:
    json.dump(recommended, f, indent=4, ensure_ascii=False)

print("\n=== Recommended capacity params ===")
print(json.dumps(recommended, indent=4))


# ============================================================
# 7. PLOTS
# ============================================================

def save_scatter_by_service(metric_x, metric_y, filename, max_points=3000):
    x_col = NODE_COLS[metric_x]
    y_col = NODE_COLS[metric_y]

    plot_df = nodes[["scenario", "service", x_col, y_col]].copy()
    plot_df = plot_df.dropna()

    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=42)

    plt.figure(figsize=(8, 6))
    for scenario in sorted(plot_df["scenario"].unique()):
        sub = plot_df[plot_df["scenario"] == scenario]
        plt.scatter(sub[x_col], sub[y_col], s=8, alpha=0.35, label=scenario)

    plt.xlabel(metric_x)
    plt.ylabel(metric_y)
    plt.title(f"{metric_x} vs {metric_y}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = OUT_DIR / filename
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved plot: {path}")


save_scatter_by_service("rps", "cpu", "plot_rps_vs_cpu.png")
save_scatter_by_service("rps", "lat_p99", "plot_rps_vs_lat_p99.png")
save_scatter_by_service("rps", "error", "plot_rps_vs_error.png")

# Replicas vs node metrics
for metric in ["cpu", "lat_p99", "error"]:
    col = NODE_COLS[metric]

    agg = (
        nodes
        .groupby(["scenario", "replicas_int"])[col]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    agg.to_csv(OUT_DIR / f"replicas_vs_{metric}.csv", index=False)

    plt.figure(figsize=(8, 5))
    for scenario in sorted(agg["scenario"].unique()):
        sub = agg[agg["scenario"] == scenario]
        plt.plot(sub["replicas_int"], sub["mean"], marker="o", label=scenario)

    plt.xlabel("replicas")
    plt.ylabel(metric)
    plt.title(f"Replicas vs {metric}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUT_DIR / f"plot_replicas_vs_{metric}.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved plot: {path}")


# Edge plots
if not edges.empty:
    def save_edge_scatter(metric_x, metric_y, filename, max_points=3000):
        x_col = EDGE_COLS[metric_x]
        y_col = EDGE_COLS[metric_y]

        if x_col not in edges.columns or y_col not in edges.columns:
            return

        plot_df = edges[["scenario", x_col, y_col]].copy().dropna()

        if len(plot_df) > max_points:
            plot_df = plot_df.sample(max_points, random_state=42)

        plt.figure(figsize=(8, 6))
        for scenario in sorted(plot_df["scenario"].unique()):
            sub = plot_df[plot_df["scenario"] == scenario]
            plt.scatter(sub[x_col], sub[y_col], s=8, alpha=0.35, label=scenario)

        plt.xlabel(metric_x)
        plt.ylabel(metric_y)
        plt.title(f"{metric_x} vs {metric_y}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = OUT_DIR / filename
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"Saved plot: {path}")

    save_edge_scatter("edge_rps", "network_latency", "plot_edge_rps_vs_network_latency.png")
    save_edge_scatter("edge_rps", "edge_error", "plot_edge_rps_vs_edge_error.png")
    save_edge_scatter("payload", "network_latency", "plot_payload_vs_network_latency.png")


# ============================================================
# 8. FINAL INDEX
# ============================================================

print("\n" + "=" * 80)
print("HOÀN TẤT PHÂN TÍCH")
print("=" * 80)
print(f"Output folder: {OUT_DIR.resolve()}")
print("\nCác file quan trọng cần gửi lại:")
print("01_metric_distribution_by_scenario.csv")
print("02_node_correlations_by_service.csv")
print("03_edge_correlations.csv")
print("04_node_capacity_exponents.csv")
print("05_edge_pressure_models.csv")
print("06_recommended_capacity_params.json")
print("plot_rps_vs_cpu.png")
print("plot_rps_vs_lat_p99.png")
print("plot_rps_vs_error.png")
print("plot_replicas_vs_cpu.png")
print("plot_replicas_vs_lat_p99.png")
print("plot_replicas_vs_error.png")
print("plot_edge_rps_vs_network_latency.png")
print("plot_edge_rps_vs_edge_error.png")
print("plot_payload_vs_network_latency.png")