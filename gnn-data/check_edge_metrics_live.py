import requests
import pandas as pd

PROM = "http://192.168.120.185:30090"
NS = "online-boutique"

queries = {
    "network_latency_seconds": f'''
histogram_quantile(
  0.99,
  sum by (source_workload, destination_workload, le)
  (
    rate(
      istio_request_duration_milliseconds_bucket{{
        reporter="source",
        source_workload_namespace="{NS}",
        destination_workload_namespace="{NS}"
      }}[1m]
    )
  )
) / 1000
''',

    "payload_size_bytes": f'''
(
  sum by (source_workload, destination_workload)
  (
    rate(
      istio_request_bytes_sum{{
        reporter="source",
        source_workload_namespace="{NS}",
        destination_workload_namespace="{NS}"
      }}[1m]
    )
  )
+
  sum by (source_workload, destination_workload)
  (
    rate(
      istio_response_bytes_sum{{
        reporter="source",
        source_workload_namespace="{NS}",
        destination_workload_namespace="{NS}"
      }}[1m]
    )
  )
) / 2
''',


    "edge_request_rate_rps": f'''
sum by (source_workload, destination_workload)
(
  rate(
    istio_requests_total{{
      reporter="source",
      source_workload_namespace="{NS}",
      destination_workload_namespace="{NS}"
    }}[1m]
  )
)
''',
    "edge_error_rate_ratio": f'''
(
  sum by (source_canonical_service, destination_canonical_service)
  (
    rate(
      istio_requests_total{{
        source_workload_namespace="{NS}",
        destination_service_namespace="{NS}",
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
        source_workload_namespace="{NS}",
        destination_service_namespace="{NS}"
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
        source_workload_namespace="{NS}",
        destination_service_namespace="{NS}"
      }}[1m]
    )
  )
)
'''    
}

rows = []

for name, q in queries.items():

    r = requests.get(
        f"{PROM}/api/v1/query",
        params={"query": q},
        timeout=30
    )

    r.raise_for_status()

    data = r.json()
    result = data["data"]["result"]

    print(f"\n=== {name} ===")
    print(f"series count: {len(result)}")

    for item in result[:10]:

        metric = item["metric"]

        src = metric.get("source_workload", "unknown")
        dst = metric.get("destination_workload", "unknown")

        val = float(item["value"][1])

        rows.append({
            "metric": name,
            "src": src,
            "dst": dst,
            "value": val,
        })

        print(f"{src:30s} -> {dst:30s} {val:.6f}")

df = pd.DataFrame(rows)

print("\n====================================================")
print("SUMMARY")
print("====================================================")

if df.empty:
    print("No edge metrics returned.")
else:

    print(df.groupby("metric")["value"].agg(["count", "min", "mean", "max"]))

    required = {
        "network_latency_seconds",
        "payload_size_bytes",
        "edge_request_rate_rps",
        "edge_error_rate_ratio",
    }

    found = set(df["metric"].unique())

    missing = required - found

    print("\nMetrics found:")
    print(sorted(found))

    if missing:
        print("\nMissing:", missing)
    else:
        print("\nOK: đủ 4 edge metrics.")
