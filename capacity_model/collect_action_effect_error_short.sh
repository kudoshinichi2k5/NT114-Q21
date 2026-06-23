#!/bin/bash
set -euo pipefail

# ============================================================
# ACTION-EFFECT ERROR-FOCUSED COLLECTION
# ============================================================
# Mục tiêu:
#   - Thu thêm action-effect samples có ERR > 0.
#   - Giữ nguyên CSV schema với action_effect_pairs_final.csv để merge trực tiếp.
#   - Không thu normal phase nữa.
#   - Chỉ chạy error_injection ngắn, tập trung load=260 và fault 30/50 để tăng ERR non-zero.
#
# Output:
#   ./action_effect_data/action_effect_pairs_error_only.csv
#
# Merge sau khi chạy:
#   head -n 1 action_effect_pairs_final.csv > action_effect_pairs_merged.csv
#   tail -n +2 action_effect_pairs_final.csv >> action_effect_pairs_merged.csv
#   tail -n +2 action_effect_pairs_error_only.csv >> action_effect_pairs_merged.csv
#
# ============================================================

WORKER_IP="192.168.120.185"
TARGET="http://$WORKER_IP:30080"
PROMETHEUS="http://$WORKER_IP:30090"
NAMESPACE="online-boutique"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCUSTFILE="$PROJECT_ROOT/workload/locustfile.py"

OUTDIR="./action_effect_data"
OUTFILE="$OUTDIR/action_effect_pairs_error_only.csv"

mkdir -p "$OUTDIR"

# Giữ danh sách service giống script cũ để schema/service order không lệch.
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
# ERROR-FOCUSED CONFIG
# ============================================================
# Chỉ dùng 1 tải cao vừa đủ, tránh chạy 6 tiếng không cần thiết.
ERROR_LOAD_LEVELS=(260)

# Tăng fault injection để ERR non-zero rõ hơn.
# 30/50 tạo lỗi đủ mạnh nhưng vẫn có hai mức để dữ liệu không quá đơn điệu.
ERROR_FAULT_LEVELS=(30 50)

# Trial ngắn: khoảng 64 samples tổng.
# Leaf trial tạo 4 samples/lần.
LEAF_TRIALS_ERROR=6
CORE_TRIALS_ERROR=8

# Core service được sample có trọng số để ưu tiên service quan trọng/lỗi.
# Bash không có weighted random tiện, nên dùng mảng lặp lại.
CORE_SERVICES_WEIGHTED=(
  "frontend" "frontend" "frontend" "frontend" "frontend"
  "checkoutservice" "checkoutservice" "checkoutservice" "checkoutservice" "checkoutservice"
  "cartservice" "cartservice"
  "recommendationservice" "recommendationservice"
  "paymentservice"
)

# Đợi lâu hơn một chút để fault/scale effect thể hiện rõ trong Prometheus.
SETTLE_BEFORE=40
SETTLE_AFTER=50

METRIC_WINDOW="45s"

R_MIN=1
R_MAX=10

# Giới hạn thời gian riêng cho job phụ này.
# 1 load × 2 fault levels × (6 leaf + 8 core) × ~90s ≈ 1h nếu chạy đủ.
# Mục tiêu là 50-80 samples lỗi chất lượng cao, không phải chạy lâu.
MAX_RUNTIME_SECONDS=7200    # 2h hard limit
START_TIME=$(date +%s)

EXPECTED_HEADER="timestamp,phase,error_injected,load_level,group,service,r_old,r_new,action,effective_delta,cpu_before,rps_before,err_before,lat_before,cpu_after,rps_after,err_after,lat_after"

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
    > /tmp/locust_action_effect_error_only.log 2>&1 &

  echo "  [locust] started: users=$users spawn=$spawn pid=$!"
}

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

choose_action() {
  local r_old=$1
  local rand=$((RANDOM % 10))
  local action

  # Error-focused:
  #   - Ưu tiên scale down/up hơn hold để tạo action-effect rõ.
  #   - Vẫn giữ hold 20% để làm baseline delta=0.
  if [ "$r_old" -le "$R_MIN" ]; then
    if [ "$rand" -lt 7 ]; then action=1; else action=0; fi
  elif [ "$r_old" -ge "$R_MAX" ]; then
    if [ "$rand" -lt 7 ]; then action=-1; else action=0; fi
  else
    if [ "$rand" -lt 4 ]; then action=1
    elif [ "$rand" -lt 8 ]; then action=-1
    else action=0
    fi
  fi

  echo "$action"
}

clamp_replicas() {
  local r=$1

  if [ "$r" -lt "$R_MIN" ]; then r=$R_MIN; fi
  if [ "$r" -gt "$R_MAX" ]; then r=$R_MAX; fi

  echo "$r"
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

run_leaf_trial() {
  local phase=$1
  local error_injected=$2
  local load=$3
  local svc r_old r_new action delta before after ts

  declare -A r_olds r_news actions deltas befores

  check_time_budget
  sleep "$SETTLE_BEFORE"

  echo "  [LEAF][$phase] choose actions..."

  for svc in "${LEAF_SERVICES[@]}"; do
    r_old=$(get_current_replicas "$svc")
    action=$(choose_action "$r_old")
    r_new=$(clamp_replicas $((r_old + action)))
    delta=$((r_new - r_old))

    r_olds[$svc]=$r_old
    r_news[$svc]=$r_new
    actions[$svc]=$action
    deltas[$svc]=$delta
    befores[$svc]=$(fetch_service_metrics "$svc")

    echo "    $svc: R=$r_old->$r_new action=$action delta=$delta"
  done

  for svc in "${LEAF_SERVICES[@]}"; do
    if [ "${r_news[$svc]}" -ne "${r_olds[$svc]}" ]; then
      scale_service "$svc" "${r_news[$svc]}"
    fi
  done

  sleep "$SETTLE_AFTER"

  ts=$(date +%s)

  for svc in "${LEAF_SERVICES[@]}"; do
    after=$(fetch_service_metrics "$svc")
    echo "${ts},${phase},${error_injected},${load},leaf,${svc},${r_olds[$svc]},${r_news[$svc]},${actions[$svc]},${deltas[$svc]},${befores[$svc]},${after}" >> "$OUTFILE"
  done

  echo "  ✓ Leaf trial done: ${#LEAF_SERVICES[@]} samples"
}

run_core_trial() {
  local phase=$1
  local error_injected=$2
  local load=$3
  local svc r_old r_new action delta before after ts

  check_time_budget

  svc="${CORE_SERVICES_WEIGHTED[$RANDOM % ${#CORE_SERVICES_WEIGHTED[@]}]}"

  sleep "$SETTLE_BEFORE"

  r_old=$(get_current_replicas "$svc")
  action=$(choose_action "$r_old")
  r_new=$(clamp_replicas $((r_old + action)))
  delta=$((r_new - r_old))

  before=$(fetch_service_metrics "$svc")

  echo "  [CORE][$phase] $svc: R=$r_old->$r_new action=$action delta=$delta"

  if [ "$r_new" -ne "$r_old" ]; then
    scale_service "$svc" "$r_new"
  fi

  sleep "$SETTLE_AFTER"

  after=$(fetch_service_metrics "$svc")
  ts=$(date +%s)

  echo "${ts},${phase},${error_injected},${load},core,${svc},${r_old},${r_new},${action},${delta},${before},${after}" >> "$OUTFILE"

  echo "  ✓ Core trial done: 1 sample"
}

run_phase_for_load_and_fault() {
  local phase=$1
  local error_injected=$2
  local load=$3
  local fault_percent=$4
  local leaf_trials=$5
  local core_trials=$6
  local i

  echo ""
  echo "────────────────────────────────────────────"
  echo "PHASE=$phase | ERROR=$error_injected | LOAD=$load | FAULT=${fault_percent}%"
  echo "────────────────────────────────────────────"

  enable_error_injection "$fault_percent"
  start_locust_background "$load" 25
  sleep 35

  echo ""
  echo "── Leaf trials: $leaf_trials × 4 services ──"
  for ((i=1; i<=leaf_trials; i++)); do
    echo ""
    echo "[$(date +%H:%M:%S)] Leaf trial $i/$leaf_trials"
    run_leaf_trial "$phase" "$error_injected" "$load"
  done

  echo ""
  echo "── Core trials: $core_trials × 1 service ──"
  for ((i=1; i<=core_trials; i++)); do
    echo ""
    echo "[$(date +%H:%M:%S)] Core trial $i/$core_trials"
    run_core_trial "$phase" "$error_injected" "$load"
  done

  stop_locust
  sleep 10
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
  echo "Top non-zero err rows:"
  awk -F',' 'NR>1 && ($13+0 > 0 || $17+0 > 0) {print $0}' "$OUTFILE" | head -n 10 || true
}

main() {
  echo "======================================================"
  echo "ACTION-EFFECT ERROR-ONLY COLLECTION"
  echo "Target: $TARGET"
  echo "Prometheus: $PROMETHEUS"
  echo "Output: $OUTFILE"
  echo ""
  echo "Error loads       : ${ERROR_LOAD_LEVELS[*]}"
  echo "Fault percentages : ${ERROR_FAULT_LEVELS[*]}"
  echo "Leaf trials/error : $LEAF_TRIALS_ERROR"
  echo "Core trials/error : $CORE_TRIALS_ERROR"
  echo "Expected samples  : $(( ${#ERROR_LOAD_LEVELS[@]} * ${#ERROR_FAULT_LEVELS[@]} * (LEAF_TRIALS_ERROR * 4 + CORE_TRIALS_ERROR) ))"
  echo "Time budget       : $MAX_RUNTIME_SECONDS seconds"
  echo "======================================================"

  init_csv
  disable_error_injection
  disable_hpa

  for fault in "${ERROR_FAULT_LEVELS[@]}"; do
    for load in "${ERROR_LOAD_LEVELS[@]}"; do
      run_phase_for_load_and_fault "error_injection" 1 "$load" "$fault" "$LEAF_TRIALS_ERROR" "$CORE_TRIALS_ERROR"
    done
  done

  disable_error_injection
  print_summary
}

main
