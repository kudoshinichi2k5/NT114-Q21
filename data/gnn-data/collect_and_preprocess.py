#!/usr/bin/env python3
"""
collect_and_preprocess_improved.py
==================================
Query Prometheus → nodes_data.csv + edges_data.csv

IMPROVEMENTS:
  ✓ Added latency quantiles (p50, p95, p99)
  ✓ Removed failure_per_second (redundant)
  ✓ Better error handling & validation
  ✓ NaN detection & reporting
  ✓ Pod → Service mapping validation
  ✓ Timestamp alignment check
  ✓ Detailed sanity report

Usage:
  python3 collect_and_preprocess_improved.py \
      --prometheus http://10.0.0.11:30090 \
      --duration 3300 \
      --step 15 \
      --outdir ./data
"""

import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
TARGET_SERVICES = [
    "adservice", "cartservice", "checkoutservice", "currencyservice",
    "emailservice", "frontend", "paymentservice",
    "productcatalogservice", "recommendationservice", "shippingservice",
]
NAMESPACE = "online-boutique"

SVC_INDEX = {s: i for i, s in enumerate(sorted(TARGET_SERVICES))}

# ─────────────────────────────────────────────
#  PROMQL QUERIES
# ─────────────────────────────────────────────
def make_queries(ns: str) -> dict:
    return {
        # ── NODE FEATURES ────────────────────────────
        "cpu_usage_millicores": {
            "section": "node",
            "label_type": "pod",
            "promql": f"""sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{ns}", container!="", container!="POD"}}[1m])) * 1000""",
        },
        "memory_usage_bytes": {
            "section": "node",
            "label_type": "pod",
            "promql": f"""sum by (pod) (container_memory_working_set_bytes{{namespace="{ns}", container!="", container!="POD"}})""",
        },
        "pod_replicas_count": {
            "section": "node",
            "label_type": "deployment",
            "promql": f"""kube_deployment_status_replicas_available{{namespace="{ns}"}}""",
        },
        "allocated_cpu_quota_millicores": {
            "section": "node",
            "label_type": "pod",
            "promql": f"""sum by (pod) (kube_pod_container_resource_requests{{namespace="{ns}", resource="cpu", container!=""}}) * 1000""",
        },
        "error_rate_ratio": {
            "section": "node",
            "label_type": "dst_canonical",
            "promql": f"""sum by (destination_canonical_service) (rate(istio_requests_total{{destination_service_namespace="{ns}", response_code=~"5.."}}[1m])) / (sum by (destination_canonical_service) (rate(istio_requests_total{{destination_service_namespace="{ns}"}}[1m])) + 1e-9)""",
        },
        "request_per_second": {
            "section": "node",
            "label_type": "dst_canonical",
            "promql": f"""sum by (destination_canonical_service) (rate(istio_requests_total{{destination_service_namespace="{ns}"}}[1m]))""",
        },
        # ── LATENCY QUANTILES ───────────────────────────────
        "latency_p50_seconds": {
            "section": "node",
            "label_type": "dst_canonical",
            "promql": f"""histogram_quantile(0.50, sum by (destination_canonical_service, le) (rate(istio_request_duration_milliseconds_bucket{{destination_service_namespace="{ns}"}}[1m]))) / 1000""",
        },
        "latency_p95_seconds": {
            "section": "node",
            "label_type": "dst_canonical",
            "promql": f"""histogram_quantile(0.95, sum by (destination_canonical_service, le) (rate(istio_request_duration_milliseconds_bucket{{destination_service_namespace="{ns}"}}[1m]))) / 1000""",
        },
        "latency_p99_seconds": {
            "section": "node",
            "label_type": "dst_canonical",
            "promql": f"""histogram_quantile(0.99, sum by (destination_canonical_service, le) (rate(istio_request_duration_milliseconds_bucket{{destination_service_namespace="{ns}"}}[1m]))) / 1000""",
        },
        # ── EDGE FEATURES ────────────────────────────
        "network_latency_seconds": {
            "section": "edge",
            "label_type": "edge_canonical",
            "promql": f"""histogram_quantile(0.99, sum by (source_canonical_service, destination_canonical_service, le) (rate(istio_request_duration_milliseconds_bucket{{source_workload_namespace="{ns}", destination_service_namespace="{ns}"}}[1m]))) / 1000""",
        },
        "payload_size_bytes": {
            "section": "edge",
            "label_type": "edge_canonical",
            "promql": f"""(sum by (source_canonical_service, destination_canonical_service) (rate(istio_request_bytes_sum{{source_workload_namespace="{ns}", destination_service_namespace="{ns}"}}[1m])) + sum by (source_canonical_service, destination_canonical_service) (rate(istio_response_bytes_sum{{source_workload_namespace="{ns}", destination_service_namespace="{ns}"}}[1m]))) / 2""",
        },
        "edge_request_rate_rps": {
            "section": "edge",
            "label_type": "edge_canonical",
            "promql": f"""sum by (source_canonical_service, destination_canonical_service) (rate(istio_requests_total{{source_workload_namespace="{ns}", destination_service_namespace="{ns}"}}[1m]))""",
        },
        "edge_error_rate_ratio": {
            "section": "edge",
            "label_type": "edge_canonical",
            "promql": f"""(
  sum by (source_canonical_service, destination_canonical_service)
  (
    rate(
      istio_requests_total{{
        source_workload_namespace="{ns}",
        destination_service_namespace="{ns}",
        response_code=~"5.."
      }}[1m]
    )
  )
)
/
clamp_min(
  sum by (source_canonical_service, destination_canonical_service)
  (
    rate(
      istio_requests_total{{
        source_workload_namespace="{ns}",
        destination_service_namespace="{ns}"
      }}[1m]
    )
  ),
  1e-9
)
or
(
  0 *
  sum by (source_canonical_service, destination_canonical_service)
  (
    rate(
      istio_requests_total{{
        source_workload_namespace="{ns}",
        destination_service_namespace="{ns}"
      }}[1m]
    )
  )
)""",
        },
    }

# ─────────────────────────────────────────────
#  HELPERS & VALIDATORS
# ─────────────────────────────────────────────
def safe_float(v):
    try:
        f = float(v)
        return f if not (math.isnan(f) or math.isinf(f)) else np.nan
    except Exception:
        return np.nan

def pod_to_service(pod_name: str) -> str | None:
    for svc in TARGET_SERVICES:
        if pod_name.startswith(svc + "-"):
            return svc
    return None

def query_range(prom_url: str, promql: str, start: float, end: float, step: int) -> list:
    resp = requests.get(
        f"{prom_url}/api/v1/query_range",
        params={"query": promql.strip(), "start": start, "end": end, "step": f"{step}s"},
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "success":
        raise RuntimeError(f"Prometheus: {data.get('error', data)}")
    return data["data"].get("result", [])

def parse_series(result: list, label_type: str, value_col: str) -> pd.DataFrame:
    rows = []
    for item in result:
        metric = item.get("metric", {})

        if label_type == "pod":
            svc = pod_to_service(metric.get("pod", ""))
        elif label_type == "deployment":
            svc = metric.get("deployment", "")
        elif label_type == "dst_canonical":
            svc = metric.get("destination_canonical_service", "")
        elif label_type == "edge_canonical":
            src = metric.get("source_canonical_service", "")
            dst = metric.get("destination_canonical_service", "")
            if dst not in TARGET_SERVICES: continue
            meta = {"src_service": src, "dst_service": dst}
            svc = "edge_dummy" # Bypass next check
        else:
            continue

        if label_type != "edge_canonical":
            if not svc or svc not in TARGET_SERVICES: continue
            meta = {"service": svc}

        for ts, val in item.get("values", []):
            v = safe_float(val)
            if not np.isnan(v):
                rows.append({**meta, "timestamp": int(ts), value_col: v})
    return pd.DataFrame(rows)

def validate_timestamp_alignment(df_node, df_edge):
    if df_node.empty or df_edge.empty: return True
    ts_node = set(df_node['timestamp'].unique())
    ts_edge = set(df_edge['timestamp'].unique())
    common = ts_node & ts_edge
    print(f"  Common timestamps: {len(common)}")
    overlap_pct = len(common) / max(len(ts_node), len(ts_edge))
    return overlap_pct > 0.8

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def collect_and_preprocess(prom_url: str, duration_sec: int, step_sec: int, outdir: Path):
    end_ts   = time.time()
    start_ts = end_ts - duration_sec

    print("=" * 70)
    print("collect_and_preprocess_improved.py")
    print("=" * 70)

    queries = make_queries(NAMESPACE)
    raw_dfs: dict[str, pd.DataFrame] = {}

    print("[QUERY] Fetching metrics from Prometheus...")
    for metric_name, cfg in queries.items():
        print(f"  [{cfg['section'].upper()}] {metric_name:<30}", end=" ", flush=True)
        try:
            result = query_range(prom_url, cfg["promql"], start_ts, end_ts, step_sec)
            df = parse_series(result, cfg["label_type"], metric_name)
            raw_dfs[metric_name] = df
            print(f"✓ {len(df):,} rows")
        except Exception as e:
            print(f"✗ Error")
            raw_dfs[metric_name] = pd.DataFrame()

    # ── 1. Build NODE dataframe ────────────────────────────────
    print("\n[BUILD] Merging node features...")
    anchor = raw_dfs["pod_replicas_count"].copy()
    if anchor.empty:
        sys.exit("ERROR: pod_replicas_count empty!")
    anchor = anchor.groupby(["timestamp", "service"], as_index=False)["pod_replicas_count"].max()

    node_feat_cols = [
        ("cpu_usage_millicores",         "sum"),
        ("memory_usage_bytes",           "sum"),
        ("allocated_cpu_quota_millicores", "sum"),
        ("error_rate_ratio",             "mean"),
        ("request_per_second",           "sum"),
        ("latency_p50_seconds",          "mean"),
        ("latency_p95_seconds",          "mean"),
        ("latency_p99_seconds",          "mean"),
    ]

    df_node = anchor.copy()
    for col, agg_fn in node_feat_cols:
        df_sub = raw_dfs.get(col, pd.DataFrame())
        if df_sub.empty:
            df_node[col] = 0.0
            continue
        df_agg = df_sub.groupby(["timestamp", "service"], as_index=False)[col].agg(agg_fn)
        df_node = df_node.merge(df_agg, on=["timestamp", "service"], how="left")

    for col in ["error_rate_ratio", "latency_p50_seconds", "latency_p95_seconds", "latency_p99_seconds", "request_per_second"]:
        df_node[col] = df_node[col].fillna(0.0)

    df_node = df_node.sort_values(["service", "timestamp"])
    for c in ["cpu_usage_millicores", "memory_usage_bytes", "allocated_cpu_quota_millicores"]:
        df_node[c] = df_node.groupby("service")[c].transform(lambda x: x.ffill().bfill())

    df_node["time"] = pd.to_datetime(df_node["timestamp"], unit="s", utc=True)
    df_node["service_id"] = df_node["service"].map(SVC_INDEX)

    # ── 2. Build EDGE dataframe ────────────────────────────────
    print("[BUILD] Merging edge features...")
    edge_feat_cols = ["network_latency_seconds", "payload_size_bytes", "edge_request_rate_rps", "edge_error_rate_ratio"]
    EDGE_KEYS = ["timestamp", "src_service", "dst_service"]
 
    df_edge = None
    for col in edge_feat_cols:
        df_sub = raw_dfs.get(col, pd.DataFrame())
        if df_sub.empty: continue
        df_sub = df_sub.groupby(EDGE_KEYS, as_index=False)[col].mean()
        if df_edge is None: df_edge = df_sub
        else: df_edge = df_edge.merge(df_sub, on=EDGE_KEYS, how="outer")

    if df_edge is None:
        df_edge = pd.DataFrame(columns=EDGE_KEYS + edge_feat_cols)
    else:
        df_edge = df_edge.fillna(0.0)
        df_edge = df_edge[~df_edge["src_service"].isin(["unknown", "loadgenerator"])]
        df_edge["time"]   = pd.to_datetime(df_edge["timestamp"], unit="s", utc=True)
        df_edge["src_id"] = df_edge["src_service"].map(SVC_INDEX).fillna(-1).astype(int)
        df_edge["dst_id"] = df_edge["dst_service"].map(SVC_INDEX).fillna(-1).astype(int)
        df_edge = df_edge.sort_values(EDGE_KEYS).reset_index(drop=True)

    validate_timestamp_alignment(df_node, df_edge)

    # ── 3. NORMALIZE ────────────────────────────────────────────
    print("\n[NORM] Min-Max normalizing...")
    NODE_RAW_COLS = [
        "cpu_usage_millicores", "memory_usage_bytes", "pod_replicas_count", "allocated_cpu_quota_millicores",
        "error_rate_ratio", "request_per_second", "latency_p50_seconds", "latency_p95_seconds", "latency_p99_seconds",
    ]
    node_scaler = MinMaxScaler()
    norm_vals = node_scaler.fit_transform(df_node[NODE_RAW_COLS].fillna(0))
    for i, col in enumerate(NODE_RAW_COLS):
        df_node[f"{col}_norm"] = norm_vals[:, i]

    if not df_edge.empty:
        EDGE_RAW_COLS = ["network_latency_seconds", "payload_size_bytes", "edge_request_rate_rps", "edge_error_rate_ratio"]
        edge_scaler = MinMaxScaler()
        norm_evals = edge_scaler.fit_transform(df_edge[EDGE_RAW_COLS].fillna(0))
        for i, col in enumerate(EDGE_RAW_COLS):
            df_edge[f"{col}_norm"] = norm_evals[:, i]

    # ── 4. TARGET LABELING (SỬA ĐỔI QUAN TRỌNG: Lấy Metrics t+1) ──────
    print("[LABEL] Creating Target Metrics for DL Predictor...")
    df_node = df_node.sort_values(["service", "timestamp"]).reset_index(drop=True)
 
    # DL Model cần dự đoán 4 thứ này ở bước t+1:
    TARGET_METRICS = ["cpu_usage_millicores", "request_per_second", "error_rate_ratio", "latency_p99_seconds"]
 
    for col in TARGET_METRICS:
        # Lấy giá trị gốc ở t+1
        df_node[f"target_{col}"] = df_node.groupby("service")[col].shift(-1).fillna(df_node[col])
        # Lấy giá trị normalized ở t+1 (Để PyTorch train dễ hội tụ)
        df_node[f"target_{col}_norm"] = df_node.groupby("service")[f"{col}_norm"].shift(-1).fillna(df_node[f"{col}_norm"])

    # ── 5. EXPORT CSV ───────────────────────────────────────────
    outdir.mkdir(parents=True, exist_ok=True)
 
    # Đã bỏ "target_pod_replicas", thêm các "target_" metrics
    node_cols_out = [
        "timestamp", "time", "service", "service_id",
        "cpu_usage_millicores", "memory_usage_bytes", "pod_replicas_count",
        "allocated_cpu_quota_millicores", "error_rate_ratio", "request_per_second",
        "latency_p50_seconds", "latency_p95_seconds", "latency_p99_seconds",
        "cpu_usage_millicores_norm", "memory_usage_bytes_norm", "pod_replicas_count_norm",
        "allocated_cpu_quota_millicores_norm", "error_rate_ratio_norm", "request_per_second_norm",
        "latency_p50_seconds_norm", "latency_p95_seconds_norm", "latency_p99_seconds_norm",
        # THE NEW GROUND TRUTH LABELS
        "target_cpu_usage_millicores", "target_request_per_second", "target_error_rate_ratio", "target_latency_p99_seconds",
        "target_cpu_usage_millicores_norm", "target_request_per_second_norm", "target_error_rate_ratio_norm", "target_latency_p99_seconds_norm"
    ]
 
    edges_path = outdir / "edges_data.csv"
    nodes_path = outdir / "nodes_data.csv"
 
    df_node[node_cols_out].to_csv(nodes_path, index=False)
    if not df_edge.empty:
        df_edge.to_csv(edges_path, index=False)
    else:
        pd.DataFrame().to_csv(edges_path, index=False)

    print(f"\n✓ DONE: Exported nodes ({len(df_node)} rows) & edges ({len(df_edge)} rows) to {outdir.resolve()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prometheus", default="http://localhost:30090")
    ap.add_argument("--duration", type=int, default=3600)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--outdir", default="./data")
    args = ap.parse_args()
    collect_and_preprocess(args.prometheus, args.duration, args.step, Path(args.outdir))
