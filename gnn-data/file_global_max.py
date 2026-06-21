import pandas as pd
import glob
import numpy as np

DATA_DIR = "./dataset-v5" # Đổi lại đường dẫn nếu cần

# 1. Tìm tất cả file CSV
node_files = glob.glob(f"{DATA_DIR}/**/nodes_data.csv", recursive=True)
edge_files = glob.glob(f"{DATA_DIR}/**/edges_data.csv", recursive=True)

print(f"[*] Đang quét {len(node_files)} file Nodes và {len(edge_files)} file Edges...\n")

# 2. Định nghĩa các Metrics cần tìm
NODE_METRICS = [
    "cpu_usage_millicores", 
    "memory_usage_bytes", 
    "request_per_second", 
    "latency_p50_seconds", 
    "latency_p95_seconds", 
    "latency_p99_seconds",
    "allocated_cpu_quota_millicores"
]

EDGE_METRICS = [
    "network_latency_seconds", 
    "payload_size_bytes", 
    "edge_request_rate_rps"
]

# Các Metrics có giới hạn vật lý tuyệt đối (Hard Limits)
global_max = {
    "error_rate_ratio": 1.0,        # Max lỗi luôn là 100% (1.0)
    "edge_error_rate_ratio": 1.0,   # Max lỗi edge luôn là 100%
    "pod_replicas_count": 10.0      # Limit của bạn đang set
}

# Khởi tạo giá trị 0 cho các metrics cần tìm
for m in NODE_METRICS + EDGE_METRICS:
    if m not in global_max:
        global_max[m] = 0.0

# 3. Quét File Nodes
for f in node_files:
    df = pd.read_csv(f)
    for col in NODE_METRICS:
        if col in df.columns:
            # Lấy Max trong file
            current_max = df[col].max()
            if not np.isnan(current_max) and current_max > global_max[col]:
                global_max[col] = current_max

# 4. Quét File Edges
for f in edge_files:
    df = pd.read_csv(f)
    for col in EDGE_METRICS:
        if col in df.columns:
            current_max = df[col].max()
            if not np.isnan(current_max) and current_max > global_max[col]:
                global_max[col] = current_max

# 5. In ra kết quả (Tự động nhân thêm 1.2 cho các biến Unbounded)
print("BẠN HÃY COPY DICTIONARY NÀY VÀO CODE COLLECT & ONLINE CONTROLLER:")
print("-" * 50)
print("GLOBAL_MAX = {")

# In Hard Limits (Không nhân 1.2)
hard_limits = ["error_rate_ratio", "edge_error_rate_ratio", "pod_replicas_count"]
for k in hard_limits:
    print(f'    "{k}": {global_max[k]:.1f},')

# In Unbounded Metrics (Nhân 1.2 buffer)
for k in NODE_METRICS + EDGE_METRICS:
    if k not in hard_limits:
        val_with_buffer = global_max[k] * 1.2
        # Đảm bảo không bị lỗi chia cho 0 nếu metric trống
        val_final = val_with_buffer if val_with_buffer > 0 else 1.0 
        print(f'    "{k}": {val_final:.2f},')

print("}")
print("-" * 50)
