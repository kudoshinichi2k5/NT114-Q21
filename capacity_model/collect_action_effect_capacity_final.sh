#!/bin/bash
set -euo pipefail

# ============================================================
# ACTION-EFFECT CAPACITY FINAL COLLECTION
# ============================================================
# Mục tiêu của bản thu cuối:
#   1. Thu nhiều action-effect samples cho core services.
#   2. Ưu tiên high-load và overload để tăng tín hiệu CPU/LAT.
#   3. Không để fault injection chi phối dữ liệu latency như lần trước.
#   4. Vẫn thu một phần nhỏ ERR để tăng non-zero error rows.
#
# Vai trò sau khi merge:
#   - CPU: target chính cho learned capacity model.
#   - LAT: dùng để train/đánh giá nếu action-effect đủ tốt; nếu không, dùng làm bằng chứng chọn hybrid conservative.
#   - ERR: dùng cho phân tích hoặc model riêng sau này; không ép learned model học ERR nếu vẫn sparse.
#   - RPS: giữ làm workload demand/persistence, không xem là capacity response chính.
#
# Output:
#   ./action_effect_data/action_effect_pairs_capacity_final.csv
#
# Merge sau khi chạy:
#   cd ./action_effect_data
#   head -n 1 action_effect_pairs_merged.csv > action_effect_pairs_v3.csv
#   tail -n +2 action_effect_pairs_merged.csv >> action_effect_pairs_v3.csv
#   tail -n +2 action_effect_pairs_capacity_final.csv >> action_effect_pairs_v3.csv
#
# ============================================================

WORKER_IP="192.168.120.185"
TARGET="http://$WORKER_IP:30080"
PROMETHEUS="http://$WORKER_IP:30090"
NAMESPACE="online-boutique"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCUSTFILE="$PROJECT_ROOT/workload/locustfile.py"

OUTDIR="./action_effect_data"
OUTFILE="$OUTDIR/action_effect_pairs_capacity_final.csv"

mkdir -p "$OUTDIR"

LEAF_SERVICES=("adservice" "emailservice" "currencyservice" "shippingservice")

CORE_SERVICES=(
  "frontend"
  "cartservice"
  "checkoutservice"
  "paymentservice"
  "productcatalogservice"
  "recommendationservice"
)

ALL_SERVICES=("${LEAF_SERVICES[@]}" "${CORE_SERVICES[@]}")

# ============================================================
# FINAL DATA COLLECTION DESIGN
# ============================================================
# Phase 1 — latency_core:
#   High-load, no fault. Thu LAT/CPU action-effect tự nhiên.
#
# Phase 2 — overload_recovery:
#   Load cao hơn, no fault. Tạo trạng thái nghẽn và đo scale recovery.
#
# Phase 3 — error_injection_small:
#   Fault nhẹ 30%, tỷ trọng nhỏ. Tăng ERR samples nhưng không làm lệch LAT.
# ============================================================

LATENCY_LOAD_LEVELS=(220 260 300 340 380 420)
OVERLOAD_LOAD_LEVELS=(300 340 380)
ERROR_LOAD_LEVELS=(260 300)

CORE_TRIALS_LATENCY=50
CORE_TRIALS_OVERLOAD=40
CORE_TRIALS_ERROR=20

ERROR_FAULT_PERCENT=30

SETTLE_BEFORE=45
SETTLE_AFTER=60
METRIC_WINDOW="45s"

R_MIN=1
R_MAX=10

# Hard limit cho chạy qua đêm. Nếu chạy xong sớm sẽ tự kết thúc.
MAX_RUNTIME_SECONDS=39600   # 11h
START_TIME=$(date +%s)

EXPECTED_HEADER="timestamp,phase,error_injected,load_level,group,service,r_old,r_new,action,effective_delta,cpu_before,rps_before,err_before,lat_before,cpu_after,rps_after,err_after,lat_after"

# Core-service weighted sampling.
# Tăng xác suất cho các service nằm trên request path chính.
CORE_SERVICES_WEIGHTED=(
  "frontend" "frontend" "frontend" "frontend" "frontend"
  "checkoutservice" "checkoutservice" "checkoutservice" "checkoutservice"
  "cartservice" "cartservice" "cartservice" "cartservice"
  "productcatalogservice" "productcatalogservice" "productcatalogservice" "productcatalogservice"
  "recommendationservice" "recommendationservice" "recommendationservice" "recommendationservice"
  "paymentservice" "paymentservice"
)

# Error phase tập trung nơi lỗi biểu hiện rõ nhất nhưng vẫn có checkout/cart/recommendation.
ERROR_SERVICES_WEIGHTED=(
  "frontend" "frontend" "frontend" "frontend" "frontend" "frontend"
  "checkoutservice" "checkoutservice" "checkoutservice"
  "cartservice" "cartservice"
  "recommendationservice" "recommendationservice"
)

# ============================================================
# CLEANUP / SETUP
# ============================================================

stop_locust() {
  pkill -f "locust" 2>/dev/null || true
  sleep 2
}

restore_hpa() {
  echo "[CLEANUP] Restore HPA..."
  for svc in "${ALL_SERVICES[@]}"; do
    kubectl autoscale deployment "$svc" -n "$NAMESPACE" \
      --cpu-percent=70 --min="$R_MIN" --max=5 \
      >/dev/null 2>&1 || true
  done
}

disable_hpa() {
  echo "[SETUP] Delete HPA..."
  for svc in "${ALL_SERVICES[@]}"; do
    kubectl delete hpa "$svc" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
  done
  sleep 5
}

disable_error_injection() {
  kubectl delete virtualservice checkout-error-injection -n "$NAMESPACE" \
    --ignore-not-found >/dev/null 2>&1 || true
}

enable_error_injection() {
  local percent="$1"

  echo "[SETUP] Enable Istio fault injection: checkoutservice ${percent}% 503"

  cat <<EOF | kubectl apply -n "$NAMESPACE" -f - >/dev/null
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: checkout-error-injection
spec:
  hosts:
  - checkoutservice
  - checkoutservice.online-boutique.svc.cluster.local
  http:
  - fault:
      abort:
        percentage:
          value: ${percent}
        httpStatus: 503
    route:
    - destination:
        host: checkoutservice
        port:
          number: 5050
EOF

  sleep 8
}

cleanup() {
  echo ""
  echo "[CLEANUP] Stop Locust, disable fault injection, restore HPA..."
  stop_locust
  disable_error_injection
  restore_hpa
}

trap cleanup EXIT INT TERM

check_time_budget() {
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - START_TIME))

  if [ "$elapsed" -ge "$MAX_RUNTIME_SECONDS" ]; then
    echo ""
    echo "[STOP] Reached time budget: ${elapsed}s >= ${MAX_RUNTIME_SECONDS}s"
    echo "[STOP] Ending collection safely."
    exit 0
  fi
}

start_locust_background() {
  local users=$1
  local spawn=$2

  stop_locust

  nohup locust -f "$LOCUSTFILE" \
    --host "$TARGET" \
    --headless \
    --users "$users" \
    --spawn-rate "$spawn" \
    --run-time 0 \
    --loglevel WARNING \
    > /tmp/locust_action_effect_capacity_final.log 2>&1 &

  echo "  [locust] started: users=$users spawn=$spawn pid=$!"
}

# ============================================================
# K8S / PROMETHEUS HELPERS
# ============================================================

get_current_replicas() {
  local svc=$1
  local replicas

  replicas=$(kubectl get deployment "$svc" -n "$NAMESPACE" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null || true)

  echo "${replicas:-1}"
}

scale_service() {
  local svc=$1
  local replicas=$2

  kubectl scale deployment "$svc" -n "$NAMESPACE" --replicas="$replicas" \
    >/dev/null 2>&1
}

prom_instant() {
  local promql="$1"

  curl -s -G "$PROMETHEUS/api/v1/query" \
    --data-urlencode "query=${promql}" \
    | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    r = d.get('data', {}).get('result', [])
    print('0' if not r else sum(float(x['value'][1]) for x in r))
except Exception:
    print('0')
"
}

fetch_service_metrics() {
  local svc=$1
  local cpu rps err lat

  cpu=$(prom_instant "sum(rate(container_cpu_usage_seconds_total{namespace=\"${NAMESPACE}\",pod=~\"${svc}-.*\",container!=\"\",container!=\"POD\"}[${METRIC_WINDOW}]))*1000")

  rps=$(prom_instant "sum(rate(istio_requests_total{destination_service_namespace=\"${NAMESPACE}\",destination_canonical_service=\"${svc}\"}[${METRIC_WINDOW}]))")

  err=$(prom_instant "sum(rate(istio_requests_total{destination_service_namespace=\"${NAMESPACE}\",destination_canonical_service=\"${svc}\",response_code=~\"5..\"}[${METRIC_WINDOW}]))/(sum(rate(istio_requests_total{destination_service_namespace=\"${NAMESPACE}\",destination_canonical_service=\"${svc}\"}[${METRIC_WINDOW}]))+0.000001)")

  lat=$(prom_instant "histogram_quantile(0.99,sum by (le)(rate(istio_request_duration_milliseconds_bucket{destination_service_namespace=\"${NAMESPACE}\",destination_canonical_service=\"${svc}\"}[${METRIC_WINDOW}])))/1000")

  echo "${cpu},${rps},${err},${lat}"
}

clamp_replicas() {
  local r=$1

  if [ "$r" -lt "$R_MIN" ]; then r=$R_MIN; fi
  if [ "$r" -gt "$R_MAX" ]; then r=$R_MAX; fi

  echo "$r"
}

choose_action_by_mode() {
  local r_old=$1
  local mode=$2
  local rand=$((RANDOM % 100))
  local action

  if [ "$r_old" -le "$R_MIN" ]; then
    if [ "$rand" -lt 75 ]; then action=1; else action=0; fi
    echo "$action"
    return
  fi

  if [ "$r_old" -ge "$R_MAX" ]; then
    if [ "$rand" -lt 75 ]; then action=-1; else action=0; fi
    echo "$action"
    return
  fi

  case "$mode" in
    latency)
      # Balanced: đủ up/down/hold để học causal action-effect.
      if [ "$rand" -lt 40 ]; then action=1
      elif [ "$rand" -lt 80 ]; then action=-1
      else action=0
      fi
      ;;
    overload)
      # Nghiêng nhẹ về down/up để tạo nghẽn và phục hồi.
      if [ "$rand" -lt 45 ]; then action=-1
      elif [ "$rand" -lt 90 ]; then action=1
      else action=0
      fi
      ;;
    error)
      # Fault phase không scale-down quá nhiều để tránh phá cluster.
      if [ "$rand" -lt 45 ]; then action=1
      elif [ "$rand" -lt 80 ]; then action=-1
      else action=0
      fi
      ;;
    *)
      if [ "$rand" -lt 40 ]; then action=1
      elif [ "$rand" -lt 80 ]; then action=-1
      else action=0
      fi
      ;;
  esac

  echo "$action"
}

init_csv() {
  if [ -f "$OUTFILE" ]; then
    CURRENT_HEADER=$(head -n 1 "$OUTFILE")
    if [ "$CURRENT_HEADER" != "$EXPECTED_HEADER" ]; then
      BACKUP="${OUTFILE}.bak.$(date +%Y%m%d_%H%M%S)"
      echo "[WARN] Old CSV schema differs. Backup to $BACKUP"
      mv "$OUTFILE" "$BACKUP"
      echo "$EXPECTED_HEADER" > "$OUTFILE"
    fi
  else
    echo "$EXPECTED_HEADER" > "$OUTFILE"
  fi
}

pick_core_service() {
  local mode=$1
  local svc

  if [ "$mode" = "error" ]; then
    svc="${ERROR_SERVICES_WEIGHTED[$RANDOM % ${#ERROR_SERVICES_WEIGHTED[@]}]}"
  else
    svc="${CORE_SERVICES_WEIGHTED[$RANDOM % ${#CORE_SERVICES_WEIGHTED[@]}]}"
  fi

  echo "$svc"
}

run_core_trial() {
  local phase=$1
  local error_injected=$2
  local load=$3
  local mode=$4

  local svc r_old r_new action delta before after ts

  check_time_budget

  svc=$(pick_core_service "$mode")

  sleep "$SETTLE_BEFORE"

  r_old=$(get_current_replicas "$svc")
  action=$(choose_action_by_mode "$r_old" "$mode")
  r_new=$(clamp_replicas $((r_old + action)))
  delta=$((r_new - r_old))

  before=$(fetch_service_metrics "$svc")

  echo "  [CORE][$phase][$mode] $svc: R=$r_old->$r_new action=$action delta=$delta"

  if [ "$r_new" -ne "$r_old" ]; then
    scale_service "$svc" "$r_new"
  fi

  sleep "$SETTLE_AFTER"

  after=$(fetch_service_metrics "$svc")
  ts=$(date +%s)

  echo "${ts},${phase},${error_injected},${load},core,${svc},${r_old},${r_new},${action},${delta},${before},${after}" >> "$OUTFILE"

  echo "  ✓ Core trial done: 1 sample"
}

run_core_phase_for_load() {
  local phase=$1
  local error_injected=$2
  local load=$3
  local mode=$4
  local core_trials=$5
  local spawn_rate=$6
  local i

  echo ""
  echo "────────────────────────────────────────────"
  echo "PHASE=$phase | ERROR=$error_injected | LOAD=$load | MODE=$mode"
  echo "────────────────────────────────────────────"

  start_locust_background "$load" "$spawn_rate"
  sleep 45

  echo ""
  echo "── Core trials: $core_trials × 1 service ──"
  for ((i=1; i<=core_trials; i++)); do
    echo ""
    echo "[$(date +%H:%M:%S)] Core trial $i/$core_trials"
    run_core_trial "$phase" "$error_injected" "$load" "$mode"
  done

  stop_locust
  sleep 12
}

print_summary() {
  echo ""
  echo "======================================================"
  echo "DONE — saved to: $OUTFILE"
  echo "======================================================"

  echo ""
  echo "Sample count by phase:"
  cut -d, -f2 "$OUTFILE" | tail -n +2 | sort | uniq -c || true

  echo ""
  echo "Sample count by error_injected:"
  cut -d, -f3 "$OUTFILE" | tail -n +2 | sort | uniq -c || true

  echo ""
  echo "Sample count by load:"
  cut -d, -f4 "$OUTFILE" | tail -n +2 | sort | uniq -c || true

  echo ""
  echo "Sample count by group:"
  cut -d, -f5 "$OUTFILE" | tail -n +2 | sort | uniq -c || true

  echo ""
  echo "Sample count by service:"
  cut -d, -f6 "$OUTFILE" | tail -n +2 | sort | uniq -c || true

  echo ""
  echo "Sample count by effective_delta:"
  cut -d, -f10 "$OUTFILE" | tail -n +2 | sort | uniq -c || true

  echo ""
  echo "Non-zero error rows:"
  awk -F',' 'NR>1 && ($13+0 > 0 || $17+0 > 0) {c++} END {print c+0}' "$OUTFILE"

  echo ""
  echo "Rows with err_after > 0:"
  awk -F',' 'NR>1 && ($17+0 > 0) {c++} END {print c+0}' "$OUTFILE"

  echo ""
  echo "High latency rows lat_before/lat_after > 0.10:"
  awk -F',' 'NR>1 && (($14+0 > 0.10) || ($18+0 > 0.10)) {c++} END {print c+0}' "$OUTFILE"

  echo ""
  echo "Top non-zero error rows:"
  awk -F',' 'NR>1 && ($13+0 > 0 || $17+0 > 0) {print $0}' "$OUTFILE" | head -n 12 || true

  echo ""
  echo "Top high latency rows:"
  awk -F',' 'NR>1 && (($14+0 > 0.10) || ($18+0 > 0.10)) {print $0}' "$OUTFILE" | head -n 12 || true
}

main() {
  local load

  echo "======================================================"
  echo "ACTION-EFFECT CAPACITY FINAL COLLECTION"
  echo "Target     : $TARGET"
  echo "Prometheus : $PROMETHEUS"
  echo "Output     : $OUTFILE"
  echo ""
  echo "Latency loads : ${LATENCY_LOAD_LEVELS[*]}"
  echo "Overload loads: ${OVERLOAD_LOAD_LEVELS[*]}"
  echo "Error loads   : ${ERROR_LOAD_LEVELS[*]}"
  echo "Error fault   : ${ERROR_FAULT_PERCENT}%"
  echo ""
  echo "Expected samples:"
  echo "  Latency  = ${#LATENCY_LOAD_LEVELS[@]} × $CORE_TRIALS_LATENCY"
  echo "  Overload = ${#OVERLOAD_LOAD_LEVELS[@]} × $CORE_TRIALS_OVERLOAD"
  echo "  Error    = ${#ERROR_LOAD_LEVELS[@]} × $CORE_TRIALS_ERROR"
  echo "  Total    ≈ $(( ${#LATENCY_LOAD_LEVELS[@]} * CORE_TRIALS_LATENCY + ${#OVERLOAD_LOAD_LEVELS[@]} * CORE_TRIALS_OVERLOAD + ${#ERROR_LOAD_LEVELS[@]} * CORE_TRIALS_ERROR )) samples"
  echo "Time budget: $MAX_RUNTIME_SECONDS seconds"
  echo "======================================================"

  init_csv
  disable_error_injection
  disable_hpa

  # Phase 1: high-load core latency without fault.
  for load in "${LATENCY_LOAD_LEVELS[@]}"; do
    disable_error_injection
    run_core_phase_for_load "latency_core" 0 "$load" "latency" "$CORE_TRIALS_LATENCY" 25
  done

  # Phase 2: overload/recovery without fault.
  for load in "${OVERLOAD_LOAD_LEVELS[@]}"; do
    disable_error_injection
    run_core_phase_for_load "overload_recovery" 0 "$load" "overload" "$CORE_TRIALS_OVERLOAD" 30
  done

  # Phase 3: small controlled error injection.
  enable_error_injection "$ERROR_FAULT_PERCENT"

  for load in "${ERROR_LOAD_LEVELS[@]}"; do
    run_core_phase_for_load "error_injection_small" 1 "$load" "error" "$CORE_TRIALS_ERROR" 25
  done

  disable_error_injection
  print_summary
}

main
