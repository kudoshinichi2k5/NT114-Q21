#!/usr/bin/env python3
"""
autoscaler.py
=============
RL Autoscaler chạy trên Kubernetes.
Mỗi 15s: query Prometheus → GAT-GRU predict → PPO decide → kubectl scale

Cần các file:
  - gat_gru_final_v4.pt    : DL model (frozen predictor)
  - ppo_curriculum_final.zip: RL policy

Chạy trên master node:
  python3 autoscaler.py \
      --prometheus http://192.168.120.185:30090 \
      --dl-model ./gat_gru_final_v4.pt \
      --rl-model ./ppo_curriculum_final.zip \
      --namespace online-boutique \
      --dry-run    # bỏ flag này khi muốn scale thật
"""

import argparse
import logging
import math
import subprocess
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
#  CONFIG — Khớp với Training
# ─────────────────────────────────────────────────────────────
TARGET_SERVICES = sorted([
    "adservice", "cartservice", "checkoutservice", "currencyservice",
    "emailservice", "frontend", "paymentservice",
    "productcatalogservice", "recommendationservice", "shippingservice",
])
NUM_NODES   = len(TARGET_SERVICES)
N_SERVICES  = len(TARGET_SERVICES)
SVC_IDX     = {s: i for i, s in enumerate(TARGET_SERVICES)}

T_WINDOW        = 4      # số bước lịch sử cho GAT-GRU
NODE_FEAT_DIM   = 9
EDGE_FEAT_DIM   = 4      # ĐÃ SỬA: Từ 3 lên 4 (Thêm edge_error)
NUM_TARGETS     = 4      # cpu, rps, err, lat
R_MIN, R_MAX    = 1, 10
SCALE_DELAY     = 2      # bước (×15s) trước khi pod thực sự ready
COOLDOWN_STEPS  = 2
LOOP_INTERVAL   = 15     # giây
NAMESPACE       = "online-boutique"

# ─────────────────────────────────────────────────────────────
#  GLOBAL MAX NORMALIZATION (Khớp 100% với lúc Train)
# ─────────────────────────────────────────────────────────────
GLOBAL_MAX = {
    "cpu_usage_millicores": 1079.8535954449,
    "memory_usage_bytes": 289815308.288,
    "pod_replicas_count": 10.0,
    "allocated_cpu_quota_millicores": 1100.0,
    "error_rate_ratio": 1.0,
    "request_per_second": 509.5377298925,
    "latency_p50_seconds": 1.5898082230,
    "latency_p95_seconds": 5.2607839816,
    "latency_p99_seconds": 8.9442161638,
    "network_latency_seconds": 5.1130203913,
    "payload_size_bytes": 75606.4124863483,
    "edge_request_rate_rps": 471.6237319865,
    "edge_error_rate_ratio": 1.0
}

def norm(value: float, key: str) -> float:
    max_val = GLOBAL_MAX.get(key, 1.0)
    if max_val == 0.0:
        return 0.0
    return float(np.clip(value / max_val, 0.0, 1.0))

NODE_FEAT_ORDER = [
    "cpu_usage_millicores", "memory_usage_bytes", "pod_replicas_count",
    "allocated_cpu_quota_millicores", "error_rate_ratio",
    "request_per_second", "latency_p50_seconds",
    "latency_p95_seconds", "latency_p99_seconds",
]
EDGE_FEAT_ORDER = [
    "network_latency_seconds", "payload_size_bytes", 
    "edge_request_rate_rps", "edge_error_rate_ratio" # ĐÃ SỬA: Thêm feature thứ 4
]

# Deployment names trên K8s (khớp với repo AhalimZaki/Online-Boutique)
DEPLOYMENT_NAMES = {
    "adservice":             "adservice",
    "cartservice":           "cartservice",
    "checkoutservice":       "checkoutservice",
    "currencyservice":       "currencyservice",
    "emailservice":          "emailservice",
    "frontend":              "frontend",
    "paymentservice":        "paymentservice",
    "productcatalogservice": "productcatalogservice",
    "recommendationservice": "recommendationservice",
    "shippingservice":       "shippingservice",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  MODEL DEFINITIONS (Khớp hoàn toàn với bản DL v4 Final)
# ─────────────────────────────────────────────────────────────

class EdgeAwareGATLayer(nn.Module):
    def __init__(self, in_node, in_edge, out_dim, dropout=0.2):
        super().__init__()
        self.W_q = nn.Linear(in_node, out_dim, bias=False)
        self.W_k = nn.Linear(in_node, out_dim, bias=False)
        self.W_v = nn.Linear(in_node, out_dim, bias=False)
        self.W_e = nn.Linear(in_edge, out_dim, bias=False)
        self.a   = nn.Linear(3 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, E, A):
        hq = self.W_q(X).unsqueeze(2).expand(-1, -1, X.shape[1], -1)
        hk = self.W_k(X).unsqueeze(1).expand(-1, X.shape[1], -1, -1)
        he = self.W_e(E)

        e = F.leaky_relu(self.a(torch.cat([hq, hk, he], dim=-1))).squeeze(-1)
        mask = torch.where(A.unsqueeze(0) > 0, e, torch.full_like(e, -9e15))
        alpha = self.dropout(F.softmax(mask, dim=-1))
        return F.elu(torch.matmul(alpha, self.W_v(X)))


class GAT_GRU_Model(nn.Module):
    EMB_DIM = 12
    def __init__(self, hidden_dim=48, dropout=0.25):
        super().__init__()
        self.node_emb = nn.Embedding(NUM_NODES, self.EMB_DIM)
        
        # 2-Hop GAT
        self.gat1 = EdgeAwareGATLayer(NODE_FEAT_DIM + self.EMB_DIM, EDGE_FEAT_DIM, hidden_dim, dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gat2 = EdgeAwareGATLayer(hidden_dim, EDGE_FEAT_DIM, hidden_dim, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + NODE_FEAT_DIM + self.EMB_DIM, 64),
            nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(),
            nn.Linear(32, NUM_TARGETS)
        )
        self.scale = nn.Parameter(torch.ones(NUM_NODES, NUM_TARGETS))
        self.bias = nn.Parameter(torch.zeros(NUM_NODES, NUM_TARGETS))

    def forward(self, X_seq, E_seq, A):
        B, T, N, _ = X_seq.shape
        Fe = E_seq.shape[-1]

        emb = self.node_emb(torch.arange(N, device=X_seq.device)).view(1, 1, N, -1).expand(B, T, N, -1)
        X_in = torch.cat([X_seq, emb], dim=-1)

        X_flat = X_in.reshape(B * T, N, -1)
        E_flat = E_seq.reshape(B * T, N, N, Fe)

        h1 = self.ln1(self.gat1(X_flat, E_flat, A))
        A2 = torch.matmul(A, A).clamp(0.0, 1.0) # 2-Hop Matrix
        h2 = self.ln2(self.gat2(h1, E_flat, A2) + h1) # Residual

        h2 = h2.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        gru_out, _ = self.gru(h2)
        final = self.ln3(gru_out[:, -1, :])

        skip = X_seq[:, -1, :, :].reshape(B * N, -1)
        emb_skip = emb[:, -1, :, :].reshape(B * N, -1)

        out = self.fc(torch.cat([final, skip, emb_skip], dim=-1)).view(B, N, NUM_TARGETS)
        out = out * self.scale.unsqueeze(0) + self.bias.unsqueeze(0)
        
        # Output chuẩn hóa về [0, 1]
        return torch.sigmoid(out)


# ─────────────────────────────────────────────────────────────
#  PROMETHEUS QUERIES
# ─────────────────────────────────────────────────────────────

PROMQL = {
    # node — label: pod
    "cpu_usage_millicores": lambda ns: (
        f'sum by (pod)(rate(container_cpu_usage_seconds_total'
        f'{{namespace="{ns}",container!="",container!="POD"}}[1m]))*1000'
    ),
    "memory_usage_bytes": lambda ns: (
        f'sum by (pod)(container_memory_working_set_bytes'
        f'{{namespace="{ns}",container!="",container!="POD"}})'
    ),
    "pod_replicas_count": lambda ns: (
        f'kube_deployment_status_replicas_available{{namespace="{ns}"}}'
    ),
    "allocated_cpu_quota_millicores": lambda ns: (
        f'sum by (pod)(kube_pod_container_resource_requests'
        f'{{namespace="{ns}",resource="cpu",container!=""}})*1000'
    ),
    "error_rate_ratio": lambda ns: (
        f'sum by (destination_canonical_service)'
        f'(rate(istio_requests_total{{destination_service_namespace="{ns}",'
        f'response_code=~"5.."}}[1m]))'
        f'/(sum by (destination_canonical_service)'
        f'(rate(istio_requests_total{{destination_service_namespace="{ns}"}}[1m]))+1e-9)'
    ),
    "request_per_second": lambda ns: (
        f'sum by (destination_canonical_service)'
        f'(rate(istio_requests_total{{destination_service_namespace="{ns}"}}[1m]))'
    ),
    "latency_p50_seconds": lambda ns: (
        f'histogram_quantile(0.50,sum by (destination_canonical_service,le)'
        f'(rate(istio_request_duration_milliseconds_bucket'
        f'{{destination_service_namespace="{ns}"}}[1m])))/1000'
    ),
    "latency_p95_seconds": lambda ns: (
        f'histogram_quantile(0.95,sum by (destination_canonical_service,le)'
        f'(rate(istio_request_duration_milliseconds_bucket'
        f'{{destination_service_namespace="{ns}"}}[1m])))/1000'
    ),
    "latency_p99_seconds": lambda ns: (
        f'histogram_quantile(0.99,sum by (destination_canonical_service,le)'
        f'(rate(istio_request_duration_milliseconds_bucket'
        f'{{destination_service_namespace="{ns}"}}[1m])))/1000'
    ),
    # edge — label: src+dst
    "network_latency_seconds": lambda ns: (
        f'histogram_quantile(0.99,sum by (source_canonical_service,'
        f'destination_canonical_service,le)(rate('
        f'istio_request_duration_milliseconds_bucket{{'
        f'source_workload_namespace="{ns}",'
        f'destination_service_namespace="{ns}"}}[1m])))/1000'
    ),
    "payload_size_bytes": lambda ns: (
        f'(sum by (source_canonical_service,destination_canonical_service)'
        f'(rate(istio_request_bytes_sum{{source_workload_namespace="{ns}",'
        f'destination_service_namespace="{ns}"}}[1m]))'
        f'+sum by (source_canonical_service,destination_canonical_service)'
        f'(rate(istio_response_bytes_sum{{source_workload_namespace="{ns}",'
        f'destination_service_namespace="{ns}"}}[1m])))/2'
    ),
    "edge_request_rate_rps": lambda ns: (
        f'sum by (source_canonical_service,destination_canonical_service)'
        f'(rate(istio_requests_total{{source_workload_namespace="{ns}",'
        f'destination_service_namespace="{ns}"}}[1m]))'
    ),
    "edge_error_rate_ratio": lambda ns: (
        f'(sum by (source_canonical_service, destination_canonical_service)'
        f'(rate(istio_requests_total{{source_workload_namespace="{ns}",'
        f'destination_service_namespace="{ns}",response_code=~"5.."}}[1m])))'
        f'/'
        f'(clamp_min(sum by (source_canonical_service, destination_canonical_service)'
        f'(rate(istio_requests_total{{source_workload_namespace="{ns}",'
        f'destination_service_namespace="{ns}"}}[1m])), 1e-9))'
    ),
}

INVALID_SRC = {"unknown", "loadgenerator", "redis-cart",
               "prometheus", "grafana", "istio-ingressgateway"}

def _pod_to_service(pod_name: str):
    for svc in TARGET_SERVICES:
        if pod_name.startswith(svc + "-"):
            return svc
    return None


def query_instant(prom_url: str, promql: str) -> list[dict]:
    """Gọi /api/v1/query (instant) — trả về list {labels, value}."""
    try:
        r = requests.get(f"{prom_url}/api/v1/query",
                         params={"query": promql.strip()}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data["status"] != "success":
            return []
        return data["data"]["result"]
    except Exception as e:
        log.warning(f"Prometheus query failed: {e}")
        return []


def fetch_metrics_snapshot(prom_url: str, ns: str) -> tuple[dict, dict]:
    """
    Lấy metrics tại thời điểm hiện tại.
    Returns:
      node_vals: {service: {feature: float}}
      edge_vals: {(src,dst): {feature: float}}
    """
    svc_set = set(TARGET_SERVICES)

    # ── Node features ────────────────────────────────────────────
    node_raw: dict[str, dict[str, float]] = {s: {} for s in TARGET_SERVICES}

    for feat in NODE_FEAT_ORDER:
        result = query_instant(prom_url, PROMQL[feat](ns))
        accum: dict[str, list[float]] = {s: [] for s in TARGET_SERVICES}

        for item in result:
            m   = item["metric"]
            val = float(item["value"][1])
            if math.isnan(val) or math.isinf(val):
                continue

            if feat == "pod_replicas_count":
                svc = m.get("deployment", "")
            elif feat in ("cpu_usage_millicores", "memory_usage_bytes",
                          "allocated_cpu_quota_millicores"):
                svc = _pod_to_service(m.get("pod", ""))
            else:
                svc = m.get("destination_canonical_service", "")

            if svc in svc_set:
                accum[svc].append(val)

        for svc in TARGET_SERVICES:
            vals = accum[svc]
            if feat == "pod_replicas_count":
                node_raw[svc][feat] = float(max(vals)) if vals else 1.0
            elif feat in ("cpu_usage_millicores", "memory_usage_bytes",
                          "allocated_cpu_quota_millicores",
                          "request_per_second"):
                node_raw[svc][feat] = float(sum(vals)) if vals else 0.0
            else:
                node_raw[svc][feat] = (float(sum(vals) / len(vals))
                                       if vals else 0.0)

    # ── Edge features ─────────────────────────────────────────────
    edge_raw: dict[tuple, dict[str, float]] = {}

    for feat in EDGE_FEAT_ORDER:
        result = query_instant(prom_url, PROMQL[feat](ns))

        for item in result:
            m   = item["metric"]
            val = float(item["value"][1])
            if math.isnan(val) or math.isinf(val):
                continue
            src = m.get("source_canonical_service", "")
            dst = m.get("destination_canonical_service", "")
            if dst not in svc_set or src in INVALID_SRC:
                continue
            key = (src, dst)
            if key not in edge_raw:
                edge_raw[key] = {}
            edge_raw[key][feat] = val

    return node_raw, edge_raw


def snapshot_to_tensors(node_raw: dict, edge_raw: dict) -> tuple:
    """
    Chuyển dict snapshot → X (N,9) và E (N,N,4) — normalized global.
    """
    N   = N_SERVICES
    X   = np.zeros((N, NODE_FEAT_DIM), dtype=np.float32)
    E   = np.zeros((N, N, EDGE_FEAT_DIM), dtype=np.float32)
    A   = np.zeros((N, N), dtype=np.float32)

    for svc, feat_map in node_raw.items():
        i = SVC_IDX[svc]
        for k, feat in enumerate(NODE_FEAT_ORDER):
            X[i, k] = norm(feat_map.get(feat, 0.0), feat)

    for (src, dst), feat_map in edge_raw.items():
        if src not in SVC_IDX or dst not in SVC_IDX:
            continue
        i, j = SVC_IDX[src], SVC_IDX[dst]
        A[i, j] = 1.0
        for k, feat in enumerate(EDGE_FEAT_ORDER):
            E[i, j, k] = norm(feat_map.get(feat, 0.0), feat)

    # Self-loops
    np.fill_diagonal(A, 1.0)

    return (torch.from_numpy(X),
            torch.from_numpy(E),
            torch.from_numpy(A))


# ─────────────────────────────────────────────────────────────
#  K8S HELPERS
# ─────────────────────────────────────────────────────────────

def get_current_replicas(ns: str) -> dict[str, int]:
    """Đọc replica count hiện tại từ K8s."""
    replicas = {}
    try:
        out = subprocess.check_output(
            ["kubectl", "get", "deployments", "-n", ns,
             "-o", "jsonpath={range .items[*]}{.metadata.name}={.spec.replicas} {end}"],
            timeout=10
        ).decode().strip()
        for pair in out.split():
            if "=" in pair:
                name, count = pair.split("=", 1)
                # Map deployment name → service name
                for svc in TARGET_SERVICES:
                    if DEPLOYMENT_NAMES[svc] == name:
                        replicas[svc] = int(count)
    except Exception as e:
        log.warning(f"kubectl get deployments failed: {e}")
    return replicas


def kubectl_scale(svc: str, replicas: int, ns: str, dry_run: bool):
    deploy = DEPLOYMENT_NAMES[svc]
    cmd = ["kubectl", "scale", "deployment", deploy,
           f"--replicas={replicas}", "-n", ns]
    if dry_run:
        log.info(f"  [DRY-RUN] {' '.join(cmd)}")
        return
    try:
        subprocess.run(cmd, check=True, timeout=15,
                       capture_output=True, text=True)
        log.info(f"  ✓ Scaled {deploy} → {replicas} replicas")
    except subprocess.CalledProcessError as e:
        log.error(f"  ✗ Scale failed: {e.stderr}")


# ─────────────────────────────────────────────────────────────
#  MAIN INFERENCE LOOP
# ─────────────────────────────────────────────────────────────

class AutoscalerAgent:
    """
    Inference loop: mỗi LOOP_INTERVAL giây thực hiện 1 cycle.
    """

    def __init__(self, prom_url, dl_path, rl_path, ns, dry_run):
        self.prom_url = prom_url
        self.ns       = ns
        self.dry_run  = dry_run
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Load DL model ────────────────────────────────────────
        log.info(f"Loading DL model: {dl_path}")
        ckpt = torch.load(dl_path, map_location=self.device)
        self.dl_model = GAT_GRU_Model(
            hidden_dim  = 48,
            dropout     = 0.25,
        ).to(self.device)
        
        self.dl_model.load_state_dict(ckpt["model_state"])
        self.dl_model.eval()
        for p in self.dl_model.parameters():
            p.requires_grad = False
        log.info("✓ DL model loaded (frozen)")

        # ── Load RL policy ───────────────────────────────────────
        log.info(f"Loading RL policy: {rl_path}")
        from stable_baselines3 import PPO
        self.rl_policy = PPO.load(rl_path, device="cpu")
        log.info("✓ RL policy loaded")

        # ── State buffers ─────────────────────────────────────────
        # Sliding window: deque của (X_tensor, E_tensor) shape (N,F)/(N,N,Fe)
        self.x_buffer:  deque = deque(maxlen=T_WINDOW)
        self.e_buffer:  deque = deque(maxlen=T_WINDOW)

        # Adjacency (cố định từ lần đọc đầu tiên)
        self.A: torch.Tensor | None = None

        # Replica tracking
        self.effective_replicas = np.ones(N_SERVICES, dtype=np.int32)
        self.desired_replicas   = np.ones(N_SERVICES, dtype=np.int32)
        self.cooldown           = np.zeros(N_SERVICES, dtype=np.int32)
        self.pending_actions: list[dict] = []

        # ĐÃ SỬA: Hyperparams SLA Warning khớp với RL Env mới nhất
        self.lat_thr = 0.05
        self.err_thr = 0.01
        self.r_max   = R_MAX
        self._init_replicas_from_k8s()

    def _init_replicas_from_k8s(self):
        """Đọc replica count thực từ K8s khi khởi động."""
        log.info("Reading current replica counts from K8s...")
        current = get_current_replicas(self.ns)
        for svc in TARGET_SERVICES:
            r = current.get(svc, 1)
            i = SVC_IDX[svc]
            self.effective_replicas[i] = r
            self.desired_replicas[i]   = r
        log.info(f"  Replicas: {dict(zip(TARGET_SERVICES, self.effective_replicas))}")

    def _warm_up_buffer(self):
        """
        Điền đủ T_WINDOW bước vào buffer trước khi bắt đầu inference.
        Lấy T_WINDOW snapshot liên tiếp cách nhau LOOP_INTERVAL giây.
        """
        log.info(f"Warming up buffer ({T_WINDOW} steps × {LOOP_INTERVAL}s)...")
        while len(self.x_buffer) < T_WINDOW:
            node_raw, edge_raw = fetch_metrics_snapshot(self.prom_url, self.ns)
            X, E, A = snapshot_to_tensors(node_raw, edge_raw)
            self.x_buffer.append(X)
            self.e_buffer.append(E)
            if self.A is None:
                self.A = A.to(self.device)
            log.info(f"  Buffer: {len(self.x_buffer)}/{T_WINDOW}")
            if len(self.x_buffer) < T_WINDOW:
                time.sleep(LOOP_INTERVAL)

    def _build_window_tensors(self) -> tuple:
        """Stack buffer → (1, T, N, F) cho GAT-GRU."""
        X_win = torch.stack(list(self.x_buffer), dim=0).unsqueeze(0).to(self.device)
        E_win = torch.stack(list(self.e_buffer), dim=0).unsqueeze(0).to(self.device)
        return X_win, E_win

    def _predict(self) -> np.ndarray:
        """GAT-GRU forward → pred (N, 4)."""
        X_win, E_win = self._build_window_tensors()
        with torch.no_grad():
            pred = self.dl_model(X_win, E_win, self.A)[0]
        return pred.cpu().numpy()   # (N, 4)

    def _apply_scale_effect(self, pred: np.ndarray) -> tuple:
        """Giống _apply_scale_effect trong training env."""
        r_eff     = self.effective_replicas.astype(np.float32)
        cpu_hat   = np.clip(pred[:, 0], 0.0, 1.0)
        rps_hat   = np.clip(pred[:, 1], 0.0, 1.0)
        lat_scale = np.sqrt(np.maximum(r_eff / 2.0, 1.0))
        err_scale = np.power(np.maximum(r_eff / 2.0, 1.0), 0.7)
        lat_hat   = np.clip(pred[:, 3] / lat_scale, 0.0, 1.0)
        err_hat   = np.clip(pred[:, 2] / err_scale, 0.0, 1.0)
        return cpu_hat, rps_hat, err_hat, lat_hat

    def _build_obs(self, x_now: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """
        Ghép state 100 chiều: giống _build_obs() trong OnlineBoutiqueScalingEnv.
        x_now: (N,9) normalized node features tại bước hiện tại
        """
        cpu_now = x_now[:, 0]
        err_now = x_now[:, 4]
        rps_now = x_now[:, 5]
        lat_now = x_now[:, 8]
        cpu_hat, rps_hat, err_hat, lat_hat = self._apply_scale_effect(pred)
        replica_norm  = self.effective_replicas.astype(np.float32) / R_MAX
        cooldown_norm = self.cooldown.astype(np.float32) / float(COOLDOWN_STEPS)
        state = np.stack([
            replica_norm, cpu_now, rps_now, err_now, lat_now,
            cpu_hat, rps_hat, err_hat, lat_hat, cooldown_norm,
        ], axis=1)  # (N, 10)
        return np.clip(state, 0.0, 1.0).astype(np.float32).reshape(-1)   # (100,)

    def _process_pending(self):
        """Tick pending scale actions — giống env.step()."""
        for item in self.pending_actions:
            item["remaining"] -= 1
        ready = [i for i in self.pending_actions if i["remaining"] <= 0]
        if ready:
            self.effective_replicas = ready[-1]["target"].copy()
            log.info(f"  Pod changes effective: "
                     f"{dict(zip(TARGET_SERVICES, self.effective_replicas))}")
        self.pending_actions = [i for i in self.pending_actions
                                 if i["remaining"] > 0]

    def step(self):
        """1 inference cycle."""
        ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")

        # 1. Fetch metrics
        node_raw, edge_raw = fetch_metrics_snapshot(self.prom_url, self.ns)
        X, E, A = snapshot_to_tensors(node_raw, edge_raw)

        # 2. Update buffer
        self.x_buffer.append(X)
        self.e_buffer.append(E)

        # 3. Predict
        pred   = self._predict()
        x_now  = X.numpy()
        obs    = self._build_obs(x_now, pred)

        # 4. PPO decide
        action, _ = self.rl_policy.predict(obs, deterministic=True)
        # action: (N,) ∈ {0,1,2} → delta {-1,0,+1}
        raw_delta = np.array(action, dtype=np.int32) - 1
        delta     = raw_delta.copy()
        delta[self.cooldown > 0] = 0   # block cooldown

        # 5. Compute new desired replicas
        new_desired   = np.clip(self.desired_replicas + delta,
                                R_MIN, R_MAX).astype(np.int32)
        actual_delta  = new_desired - self.desired_replicas
        self.desired_replicas = new_desired

        # 6. Execute kubectl scale
        if np.any(actual_delta != 0):
            for svc in TARGET_SERVICES:
                i = SVC_IDX[svc]
                if actual_delta[i] != 0:
                    kubectl_scale(svc, int(new_desired[i]), self.ns, self.dry_run)
            self.pending_actions.append({
                "remaining": SCALE_DELAY,
                "target":    self.desired_replicas.copy(),
            })
            self.cooldown[actual_delta != 0] = COOLDOWN_STEPS

        # 7. Tick cooldown + pending
        self.cooldown = np.maximum(self.cooldown - 1, 0)
        self._process_pending()

        # 8. Log
        sla_services = []
        cpu_hat, rps_hat, err_hat, lat_hat = self._apply_scale_effect(pred)
        for i, svc in enumerate(TARGET_SERVICES):
            if lat_hat[i] > self.lat_thr or err_hat[i] > self.err_thr:
                sla_services.append(
                    f"{svc}(lat={lat_hat[i]:.2f},err={err_hat[i]:.3f})"
                )

        log.info(
            f"[{ts}] "
            f"reps={list(self.effective_replicas)} | "
            f"delta={list(actual_delta)} | "
            f"sla_warn={sla_services or 'none'}"
        )

    def run(self):
        self._warm_up_buffer()
        log.info("Starting inference loop (Ctrl-C to stop)...")
        try:
            while True:
                t0 = time.time()
                self.step()
                elapsed = time.time() - t0
                sleep_t = max(0.0, LOOP_INTERVAL - elapsed)
                time.sleep(sleep_t)
        except KeyboardInterrupt:
            log.info("Autoscaler stopped.")


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="RL Autoscaler for Online Boutique")
    ap.add_argument("--prometheus",  default="http://localhost:30090")
    ap.add_argument("--dl-model",    default="./gat_gru_final_v4.pt")
    ap.add_argument("--rl-model",    default="./ppo_curriculum_final.zip")
    ap.add_argument("--namespace",   default=NAMESPACE)
    ap.add_argument("--dry-run",     action="store_true",
                    help="Log actions but do NOT call kubectl scale")
    args = ap.parse_args()

    log.info("=" * 60)
    log.info("RL AUTOSCALER — Online Boutique")
    log.info(f"  Prometheus : {args.prometheus}")
    log.info(f"  DL model   : {args.dl_model}")
    log.info(f"  RL model   : {args.rl_model}")
    log.info(f"  Namespace  : {args.namespace}")
    log.info(f"  Dry-run    : {args.dry_run}")
    log.info("=" * 60)

    agent = AutoscalerAgent(
        prom_url = args.prometheus,
        dl_path  = args.dl_model,
        rl_path  = args.rl_model,
        ns       = args.namespace,
        dry_run  = args.dry_run,
    )
    agent.run()


if __name__ == "__main__":
    main()
