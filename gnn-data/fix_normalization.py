import pandas as pd
import glob
import numpy as np
import os

DATA_DIR = "./dataset-v5"

# Bộ số Global Max chính xác bạn vừa tính ra
GLOBAL_MAX = {
    "error_rate_ratio": 1.0,
    "edge_error_rate_ratio": 1.0,
    "pod_replicas_count": 10.0,
    "cpu_usage_millicores": 1373.43,
    "memory_usage_bytes": 318249369.60,
    "request_per_second": 688.10,
    "latency_p50_seconds": 2.61,
    "latency_p95_seconds": 11.83,
    "latency_p99_seconds": 30.52,
    "allocated_cpu_quota_millicores": 1200.00,
    "network_latency_seconds": 11.61,
    "payload_size_bytes": 105506.75,
    "edge_request_rate_rps": 668.44,
}

NODE_METRICS = [
    "cpu_usage_millicores", "memory_usage_bytes", "pod_replicas_count",
    "allocated_cpu_quota_millicores", "error_rate_ratio", "request_per_second", 
    "latency_p50_seconds", "latency_p95_seconds", "latency_p99_seconds"
]

EDGE_METRICS = [
    "network_latency_seconds", "payload_size_bytes", 
    "edge_request_rate_rps", "edge_error_rate_ratio"
]

def process_files():
    if not os.path.exists(DATA_DIR):
        print(f"LỖI: Không tìm thấy thư mục {DATA_DIR}")
        return

    # 1. Xử lý file Nodes
    node_files = glob.glob(f"{DATA_DIR}/**/nodes_data.csv", recursive=True)
    for f in node_files:
        df = pd.read_csv(f)
        modified = False
        
        for col in NODE_METRICS:
            if col in df.columns:
                max_val = GLOBAL_MAX.get(col, 1.0)
                # Tính lại cột Normalize từ cột gốc
                df[f"{col}_norm"] = np.clip(df[col] / max_val, 0.0, 1.0)
                modified = True
                
                # Tính lại cột Target Normalize từ cột Target gốc
                target_col = f"target_{col}"
                if target_col in df.columns:
                    df[f"{target_col}_norm"] = np.clip(df[target_col] / max_val, 0.0, 1.0)
        
        if modified:
            df.to_csv(f, index=False)
            print(f"[Nodes] Đã ghi đè thành công: {f}")

    # 2. Xử lý file Edges
    edge_files = glob.glob(f"{DATA_DIR}/**/edges_data.csv", recursive=True)
    for f in edge_files:
        df = pd.read_csv(f)
        modified = False
        
        for col in EDGE_METRICS:
            if col in df.columns:
                max_val = GLOBAL_MAX.get(col, 1.0)
                df[f"{col}_norm"] = np.clip(df[col] / max_val, 0.0, 1.0)
                modified = True
                
        if modified:
            df.to_csv(f, index=False)
            print(f"[Edges] Đã ghi đè thành công: {f}")

    print("\n✓ HOÀN TẤT! Dữ liệu đã được chuẩn hóa theo tỷ lệ Global.")

if __name__ == "__main__":
    process_files()
