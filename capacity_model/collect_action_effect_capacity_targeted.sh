#!/bin/bash
set -euo pipefail

# ============================================================
# ACTION-EFFECT CAPACITY TARGETED COLLECTION
# ============================================================
# Purpose:
#   Collect the final targeted action-effect dataset for Capacity Model refinement.
#
# Design principle:
#   Capacity Model is NOT a workload/anomaly generator.
#   Its role is to estimate the effect of a scaling action:
#
#       (service, r_old, r_new, action, load, metrics_before)
#           -> metrics_after
#
#   Therefore, this script focuses on controlled scale-response under high load,
#   especially for core services where RL autoscaling decisions matter most.
#
# What this collection targets:
#   - CPU action-effect: should remain the main learned target.
#   - LAT action-effect: collect cleaner high-load samples without fault injection.
#   - RPS: external demand, kept for context but not treated as capacity response.
#   - ERR: kept as observed context, but this script does not force error injection.
#
# Why no heavy fault injection here:
#   Previous data already showed that fault/dependency latency can make scaling look
#   ineffective. For Capacity Model, the remaining missing signal is clean overload
#   scale-response, not more injected failures.
#
# Output:
#   ./action_effect_data/action_effect_pairs_capacity_targeted.csv
#
# Suggested merge:
#   cd ./action_effect_data
#   head -n 1 action_effect_pairs_v3.csv > action_effect_pairs_v4.csv
#   tail -n +2 action_effect_pairs_v3.csv >> action_effect_pairs_v4.csv
#   tail -n +2 action_effect_pairs_capacity_targeted.csv >> action_effect_pairs_v4.csv
# ============================================================

WORKER_IP="192.168.120.185"
TARGET="http://$WORKER_IP:30080"
PROMETHEUS="http://$WORKER_IP:30090"
NAMESPACE="online-boutique"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCUSTFILE="$PROJECT_ROOT/workload/locustfile.py"

OUTDIR="./action_effect_data"
OUTFILE="$OUTDIR/action_effect_pairs_capacity_targeted.csv"

mkdir -p "$OUTDIR"

CORE_SERVICES=(
  "frontend"
  "cartservice"
  "checkoutservice"
  "paymentservice"
  "productcatalogservice"
  "recommendationservice"
)

ALL_SERVICES=(
  "adservice"
  "cartservice"
  "checkoutservice"
  "currencyservice"
  "emailservice"
  "frontend"
  "paymentservice"
  "productcatalogservice"
  "recommendationservice"
  "shippingservice"
)

# High-load, no-fault levels. These complement action_effect_pairs_v3:
# - avoid low/normal loads already overrepresented
# - focus pressure zones where latency response is meaningful
LOAD_LEVELS=(300 340 380 420)

# 4 loads × 36 trials = 144 samples.
# With settle_before=45 and settle_after=90, runtime is roughly 5.5-6.5h.
TRIALS_PER_LOAD=36

SETTLE_BEFORE=45
SETTLE_AFTER=90
METRIC_WINDOW="45s"

R_MIN=1
R_MAX=10

MAX_RUNTIME_SECONDS=25200   # 7h
START_TIME=$(date +%s)

EXPECTED_HEADER="timestamp,phase,error_injected,load_level,group,service,r_old,r_new,action,effective_delta,cpu_before,rps_before,err_before,lat_before,cpu_after,rps_after,err_after,lat_after"

# Weighted sampling:
# frontend/checkout/cart are emphasized because prior validation showed these
# services have the largest latency uncertainty and directly affect SLA.
CORE_SERVICES_WEIGHTED=(
  "frontend" "frontend" "frontend" "frontend" "frontend" "frontend"
  "checkoutservice" "checkoutservice" "checkoutservice" "checkoutservice" "checkoutservice"
  "cartservice" "cartservice" "cartservice" "cartservice"
  "productcatalogservice" "productcatalogservice" "productcatalogservice"
  "recommendationservice" "recommendationservice" "recommendationservice"
  "paymentservice" "paymentservice"
)

stop_locust() {
  pkill -f "locust" 2>/dev/null || true
  sleep 2
}

disable_error_injection() {
  kubectl delete virtualservice checkout-error-injection -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
  kubectl delete virtualservice productcatalogservice-v5-fault -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
  kubectl delete virtualservice cartservice-v5-fault -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
  kubectl delete virtualservice paymentservice-v5-fault -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
  kubectl delete virtualservice checkoutservice-v5-fault -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
  kubectl delete virtualservice recommendationservice-v5-fault -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
}

disable_hpa() {
  echo "[SETUP] Delete HPA..."
  for svc in "${ALL_SERVICES[@]}"; do
    kubectl delete hpa "$svc" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
  done
  sleep 5
}

restore_hpa() {
  echo "[CLEANUP] Restore HPA..."
  for svc in "${ALL_SERVICES[@]}"; do
    kubectl autoscale deployment "$svc" -n "$NAMESPACE" \
      --cpu-percent=70 --min="$R_MIN" --max=5 \
      >/dev/null 2>&1 || true
  done
}

cleanup() {
  echo ""
  echo "[CLEANUP] Stop Locust, disable faults, restore HPA..."
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
    > /tmp/locust_action_effect_capacity_targeted.log 2>&1 &

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

  kubectl scale deployment "$svc" -n "$NAMESPACE" --replicas="$replicas" >/dev/null 2>&1
}

set_core_replicas() {
  local replicas=$1
  local svc

  echo "[SETUP] Set core services to ${replicas} replicas"
  for svc in "${CORE_SERVICES[@]}"; do
    scale_service "$svc" "$replicas"
  done
  sleep 45
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

pick_core_service() {
  local svc
  svc="${CORE_SERVICES_WEIGHTED[$RANDOM % ${#CORE_SERVICES_WEIGHTED[@]}]}"
  echo "$svc"
}

choose_base_replicas() {
  local load=$1
  local r

  if [ "$load" -ge 400 ]; then
    r=$((1 + RANDOM % 6))    # 1..6
  elif [ "$load" -ge 360 ]; then
    r=$((1 + RANDOM % 5))    # 1..5
  else
    r=$((1 + RANDOM % 4))    # 1..4
  fi

  echo "$r"
}

choose_action_balanced() {
  local r_old=$1
  local rand=$((RANDOM % 100))
  local action

  # Match RL action space: action delta in {-1,0,+1}.
  # Do not collect +/-2 because the current PPO env uses MAX_DELTA=1.
  if [ "$r_old" -le "$R_MIN" ]; then
    if [ "$rand" -lt 75 ]; then action=1; else action=0; fi
  elif [ "$r_old" -ge "$R_MAX" ]; then
    if [ "$rand" -lt 75 ]; then action=-1; else action=0; fi
  else
    # 45% up, 40% down, 15% hold.
    if [ "$rand" -lt 45 ]; then action=1
    elif [ "$rand" -lt 85 ]; then action=-1
    else action=0
    fi
  fi

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

run_targeted_trial() {
  local load=$1
  local svc r_base r_old r_new action delta before after ts

  check_time_budget

  svc=$(pick_core_service)
  r_base=$(choose_base_replicas "$load")
  r_base=$(clamp_replicas "$r_base")

  echo "  [TARGETED] prepare $svc at R=$r_base"
  scale_service "$svc" "$r_base"

  sleep "$SETTLE_BEFORE"

  r_old=$(get_current_replicas "$svc")
  action=$(choose_action_balanced "$r_old")
  r_new=$(clamp_replicas $((r_old + action)))
  delta=$((r_new - r_old))

  before=$(fetch_service_metrics "$svc")

  echo "  [TARGETED] load=$load | $svc: R=$r_old->$r_new action=$action delta=$delta"

  if [ "$r_new" -ne "$r_old" ]; then
    scale_service "$svc" "$r_new"
  fi

  sleep "$SETTLE_AFTER"

  after=$(fetch_service_metrics "$svc")
  ts=$(date +%s)

  echo "${ts},targeted_capacity,0,${load},core,${svc},${r_old},${r_new},${action},${delta},${before},${after}" >> "$OUTFILE"

  echo "  ✓ Targeted trial done: 1 sample"
}

run_phase_for_load() {
  local load=$1
  local i

  echo ""
  echo "────────────────────────────────────────────"
  echo "PHASE=targeted_capacity | ERROR=0 | LOAD=$load"
  echo "────────────────────────────────────────────"

  disable_error_injection
  set_core_replicas 2

  start_locust_background "$load" 30
  sleep 60

  echo ""
  echo "── Targeted core trials: $TRIALS_PER_LOAD × 1 service ──"
  for ((i=1; i<=TRIALS_PER_LOAD; i++)); do
    echo ""
    echo "[$(date +%H:%M:%S)] Targeted trial $i/$TRIALS_PER_LOAD"
    run_targeted_trial "$load"
  done

  stop_locust
  sleep 15
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
  echo "Sample count by service:"
  cut -d, -f6 "$OUTFILE" | tail -n +2 | sort | uniq -c || true

  echo ""
  echo "Sample count by effective_delta:"
  cut -d, -f10 "$OUTFILE" | tail -n +2 | sort | uniq -c || true

  echo ""
  echo "Non-zero error rows:"
  awk -F',' 'NR>1 && ($13+0 > 0 || $17+0 > 0) {c++} END {print c+0}' "$OUTFILE"

  echo ""
  echo "High latency rows lat_before/lat_after > 0.10:"
  awk -F',' 'NR>1 && (($14+0 > 0.10) || ($18+0 > 0.10)) {c++} END {print c+0}' "$OUTFILE"

  echo ""
  echo "Scale-up samples:"
  awk -F',' 'NR>1 && ($10+0 > 0) {c++} END {print c+0}' "$OUTFILE"

  echo ""
  echo "Scale-down samples:"
  awk -F',' 'NR>1 && ($10+0 < 0) {c++} END {print c+0}' "$OUTFILE"

  echo ""
  echo "Hold samples:"
  awk -F',' 'NR>1 && ($10+0 == 0) {c++} END {print c+0}' "$OUTFILE"

  echo ""
  echo "Top high latency rows:"
  awk -F',' 'NR>1 && (($14+0 > 0.10) || ($18+0 > 0.10)) {print $0}' "$OUTFILE" | head -n 12 || true
}

main() {
  local load

  echo "======================================================"
  echo "ACTION-EFFECT CAPACITY TARGETED COLLECTION"
  echo "Target     : $TARGET"
  echo "Prometheus : $PROMETHEUS"
  echo "Output     : $OUTFILE"
  echo ""
  echo "Loads      : ${LOAD_LEVELS[*]}"
  echo "Trials/load: $TRIALS_PER_LOAD"
  echo "Total      : $(( ${#LOAD_LEVELS[@]} * TRIALS_PER_LOAD )) samples"
  echo "Faults     : disabled"
  echo "Time budget: $MAX_RUNTIME_SECONDS seconds"
  echo "======================================================"

  init_csv
  disable_error_injection
  disable_hpa

  for load in "${LOAD_LEVELS[@]}"; do
    run_phase_for_load "$load"
  done

  disable_error_injection
  print_summary
}

main
