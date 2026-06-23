#!/bin/bash
set -euo pipefail

# ============================================================
# ACTION-EFFECT FRONTEND ERROR TOP-UP COLLECTION
# ============================================================
# Mục tiêu:
#   - Thu thêm ERR non-zero tập trung vào frontend.
#   - Giữ nguyên CSV schema với action_effect_pairs_final.csv để merge trực tiếp.
#   - Chỉ chạy error_injection, load=260, fault 50/70.
#
# Output:
#   ./action_effect_data/action_effect_pairs_frontend_error_topup.csv
# ============================================================

WORKER_IP="192.168.120.185"
TARGET="http://$WORKER_IP:30080"
PROMETHEUS="http://$WORKER_IP:30090"
NAMESPACE="online-boutique"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCUSTFILE="$PROJECT_ROOT/workload/locustfile.py"

OUTDIR="./action_effect_data"
OUTFILE="$OUTDIR/action_effect_pairs_frontend_error_topup.csv"
mkdir -p "$OUTDIR"

LEAF_SERVICES=("adservice" "emailservice" "currencyservice" "shippingservice")
CORE_SERVICES=("frontend" "cartservice" "checkoutservice" "paymentservice" "productcatalogservice" "recommendationservice")
ALL_SERVICES=("${LEAF_SERVICES[@]}" "${CORE_SERVICES[@]}")

ERROR_LOAD_LEVELS=(260)
ERROR_FAULT_LEVELS=(50 70)
CORE_TRIALS_ERROR=18

SETTLE_BEFORE=40
SETTLE_AFTER=55
METRIC_WINDOW="45s"

R_MIN=1
R_MAX=10
MAX_RUNTIME_SECONDS=5400
START_TIME=$(date +%s)

EXPECTED_HEADER="timestamp,phase,error_injected,load_level,group,service,r_old,r_new,action,effective_delta,cpu_before,rps_before,err_before,lat_before,cpu_after,rps_after,err_after,lat_after"

# Lần thu trước non-zero ERR chỉ xuất hiện ở frontend, nên ưu tiên frontend.
CORE_SERVICES_WEIGHTED=(
  "frontend" "frontend" "frontend" "frontend" "frontend" "frontend" "frontend" "frontend"
  "checkoutservice" "checkoutservice"
  "recommendationservice"
)

stop_locust() {
  pkill -f "locust" 2>/dev/null || true
  sleep 2
}

restore_hpa() {
  echo "[CLEANUP] Restore HPA..."
  for svc in "${ALL_SERVICES[@]}"; do
    kubectl autoscale deployment "$svc" -n "$NAMESPACE" --cpu-percent=70 --min="$R_MIN" --max=5 >/dev/null 2>&1 || true
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
  kubectl delete virtualservice checkout-error-injection -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
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
    echo "[STOP] Reached time budget: ${elapsed}s >= ${MAX_RUNTIME_SECONDS}s"
    exit 0
  fi
}

start_locust_background() {
  local users=$1
  local spawn=$2
  stop_locust
  nohup locust -f "$LOCUSTFILE" --host "$TARGET" --headless --users "$users" --spawn-rate "$spawn" --run-time 0 --loglevel WARNING > /tmp/locust_frontend_error_topup.log 2>&1 &
  echo "  [locust] started: users=$users spawn=$spawn pid=$!"
}

get_current_replicas() {
  local svc=$1
  local replicas
  replicas=$(kubectl get deployment "$svc" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || true)
  echo "${replicas:-1}"
}

scale_service() {
  local svc=$1
  local replicas=$2
  kubectl scale deployment "$svc" -n "$NAMESPACE" --replicas="$replicas" >/dev/null 2>&1
}

prom_instant() {
  local promql="$1"
  curl -s -G "$PROMETHEUS/api/v1/query" --data-urlencode "query=${promql}" | python3 -c "
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

  if [ "$r_old" -le "$R_MIN" ]; then
    if [ "$rand" -lt 8 ]; then action=1; else action=0; fi
  elif [ "$r_old" -ge "$R_MAX" ]; then
    if [ "$rand" -lt 8 ]; then action=-1; else action=0; fi
  else
    if [ "$rand" -lt 5 ]; then action=1
    elif [ "$rand" -lt 9 ]; then action=-1
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
  local core_trials=$5
  local i

  echo ""
  echo "────────────────────────────────────────────"
  echo "PHASE=$phase | ERROR=$error_injected | LOAD=$load | FAULT=${fault_percent}%"
  echo "────────────────────────────────────────────"

  enable_error_injection "$fault_percent"
  start_locust_background "$load" 25
  sleep 35

  echo ""
  echo "── Core error top-up trials: $core_trials × 1 service ──"
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
  echo "ACTION-EFFECT FRONTEND ERROR TOP-UP COLLECTION"
  echo "Target: $TARGET"
  echo "Prometheus: $PROMETHEUS"
  echo "Output: $OUTFILE"
  echo ""
  echo "Error loads       : ${ERROR_LOAD_LEVELS[*]}"
  echo "Fault percentages : ${ERROR_FAULT_LEVELS[*]}"
  echo "Core trials/error : $CORE_TRIALS_ERROR"
  echo "Expected samples  : $(( ${#ERROR_LOAD_LEVELS[@]} * ${#ERROR_FAULT_LEVELS[@]} * CORE_TRIALS_ERROR ))"
  echo "Time budget       : $MAX_RUNTIME_SECONDS seconds"
  echo "======================================================"

  init_csv
  disable_error_injection
  disable_hpa

  for fault in "${ERROR_FAULT_LEVELS[@]}"; do
    for load in "${ERROR_LOAD_LEVELS[@]}"; do
      run_phase_for_load_and_fault "error_injection" 1 "$load" "$fault" "$CORE_TRIALS_ERROR"
    done
  done

  disable_error_injection
  print_summary
}

main
