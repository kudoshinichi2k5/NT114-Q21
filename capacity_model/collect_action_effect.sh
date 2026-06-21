#!/bin/bash
# ============================================================
# collect_action_effect.sh
# ============================================================
# Thu thập dữ liệu action-effect thật:
# (r_old, r_new, action, metrics_before, load)
#        -> metrics_after
#
# Dùng để train Learned Capacity Model cho Offline RL.
# Script sẽ tắt HPA trước khi chạy và khôi phục lại khi thoát.
# ============================================================

set -euo pipefail

WORKER_IP="192.168.120.185"
TARGET="http://$WORKER_IP:30080"
PROMETHEUS="http://$WORKER_IP:30090"
NAMESPACE="online-boutique"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCUSTFILE="$PROJECT_ROOT/workload/locustfile.py"

OUTDIR="./action_effect_data"
OUTFILE="$OUTDIR/action_effect_pairs.csv"

mkdir -p "$OUTDIR"

SERVICES=(
    "frontend"
    "checkoutservice"
    "currencyservice"
    "cartservice"
    "productcatalogservice"
    "recommendationservice"
    "shippingservice"
    "adservice"
    "emailservice"
    "paymentservice"
)

LOAD_LEVELS=(40 80 120 160)

TRIALS_PER_LOAD=40

SETTLE_BEFORE=60
SETTLE_AFTER=75

R_MIN=1
R_MAX=10

EXPECTED_HEADER="timestamp,load_level,service,r_old,r_new,action,effective_delta,cpu_before,rps_before,err_before,lat_before,cpu_after,rps_after,err_after,lat_after"

# ============================================================
# CLEANUP
# ============================================================

stop_locust() {
    pkill -f "locust" 2>/dev/null || true
    sleep 2
}

restore_hpa() {
    echo "[CLEANUP] Khôi phục lại HPA cho tất cả services..."
    for svc in "${SERVICES[@]}"; do
        kubectl autoscale deployment "$svc" -n "$NAMESPACE" \
            --cpu-percent=70 --min="$R_MIN" --max="$R_MAX" \
            >/dev/null 2>&1 || true
    done
    echo "  ✓ HPA restored"
}

cleanup() {
    echo ""
    echo "[CLEANUP] Dừng Locust và khôi phục HPA..."
    stop_locust
    restore_hpa
}

trap cleanup EXIT

# ============================================================
# HELPERS
# ============================================================

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
        > /tmp/locust_action_effect.log 2>&1 &

    echo "  [locust] started: $users users (PID $!)"
}

disable_hpa() {
    echo "[SETUP] Tắt HPA tất cả services..."
    for svc in "${SERVICES[@]}"; do
        kubectl delete hpa "$svc" -n "$NAMESPACE" --ignore-not-found || true
    done
    sleep 5
}

get_current_replicas() {
    local svc=$1
    local replicas

    replicas=$(kubectl get deployment "$svc" -n "$NAMESPACE" \
        -o jsonpath='{.spec.replicas}' 2>/dev/null || true)

    if [[ -z "$replicas" ]]; then
        echo "1"
    else
        echo "$replicas"
    fi
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
    if not r:
        print('0')
    else:
        print(sum(float(x['value'][1]) for x in r))
except Exception:
    print('0')
"
}

fetch_service_metrics() {
    local svc=$1
    local cpu rps err lat

    cpu=$(prom_instant "sum(rate(container_cpu_usage_seconds_total{namespace=\"${NAMESPACE}\",pod=~\"${svc}-.*\",container!=\"\",container!=\"POD\"}[1m]))*1000")

    rps=$(prom_instant "sum(rate(istio_requests_total{destination_service_namespace=\"${NAMESPACE}\",destination_canonical_service=\"${svc}\"}[1m]))")

    err=$(prom_instant "sum(rate(istio_requests_total{destination_service_namespace=\"${NAMESPACE}\",destination_canonical_service=\"${svc}\",response_code=~\"5..\"}[1m]))/(sum(rate(istio_requests_total{destination_service_namespace=\"${NAMESPACE}\",destination_canonical_service=\"${svc}\"}[1m]))+0.000001)")

    lat=$(prom_instant "histogram_quantile(0.99,sum by (le)(rate(istio_request_duration_milliseconds_bucket{destination_service_namespace=\"${NAMESPACE}\",destination_canonical_service=\"${svc}\"}[1m])))/1000")

    echo "${cpu},${rps},${err},${lat}"
}

choose_action() {
    local r_old=$1
    local rand=$((RANDOM % 10))
    local action

    if [ "$r_old" -le "$R_MIN" ]; then
        # Ở min pod thì tránh scale down vô hiệu quá nhiều
        if [ "$rand" -lt 7 ]; then
            action=1
        else
            action=0
        fi
    elif [ "$r_old" -ge "$R_MAX" ]; then
        # Ở max pod thì tránh scale up vô hiệu quá nhiều
        if [ "$rand" -lt 7 ]; then
            action=-1
        else
            action=0
        fi
    else
        # Ở giữa giữ phân phối tương đối tự nhiên
        if [ "$rand" -lt 4 ]; then
            action=1
        elif [ "$rand" -lt 8 ]; then
            action=-1
        else
            action=0
        fi
    fi

    echo "$action"
}

# ============================================================
# INIT CSV
# ============================================================

if [ -f "$OUTFILE" ]; then
    CURRENT_HEADER=$(head -n 1 "$OUTFILE")
    if [ "$CURRENT_HEADER" != "$EXPECTED_HEADER" ]; then
        BACKUP="${OUTFILE}.bak.$(date +%Y%m%d_%H%M%S)"
        echo "[WARN] CSV cũ khác schema, backup sang: $BACKUP"
        mv "$OUTFILE" "$BACKUP"
        echo "$EXPECTED_HEADER" > "$OUTFILE"
    fi
else
    echo "$EXPECTED_HEADER" > "$OUTFILE"
fi

# ============================================================
# MAIN
# ============================================================

echo "======================================================"
echo " ACTION-EFFECT DATA COLLECTION"
echo " Target      : $TARGET"
echo " Prometheus  : $PROMETHEUS"
echo " Output      : $OUTFILE"
echo " Load levels : ${LOAD_LEVELS[*]}"
echo " Trials/load : $TRIALS_PER_LOAD"
echo " R_MIN/R_MAX : $R_MIN / $R_MAX"
echo "======================================================"

disable_hpa

TOTAL_TRIALS=$(( ${#LOAD_LEVELS[@]} * TRIALS_PER_LOAD ))
TRIAL_NUM=0

for load in "${LOAD_LEVELS[@]}"; do

    echo ""
    echo "─────────────────────────────────────────────────────"
    echo " LOAD LEVEL: $load users"
    echo "─────────────────────────────────────────────────────"

    start_locust_background "$load" 10
    sleep 30

    for ((i=1; i<=TRIALS_PER_LOAD; i++)); do
        TRIAL_NUM=$((TRIAL_NUM + 1))

        echo ""
        echo "[$(date +%H:%M:%S)] Trial $TRIAL_NUM/$TOTAL_TRIALS (load=$load)"

        sleep "$SETTLE_BEFORE"

        svc="${SERVICES[$RANDOM % ${#SERVICES[@]}]}"
        r_old=$(get_current_replicas "$svc")

        action=$(choose_action "$r_old")

        r_new=$(( r_old + action ))

        if [ "$r_new" -lt "$R_MIN" ]; then
            r_new=$R_MIN
        fi

        if [ "$r_new" -gt "$R_MAX" ]; then
            r_new=$R_MAX
        fi

        effective_delta=$(( r_new - r_old ))

        echo "  Service: $svc | R_old=$r_old -> R_new=$r_new (action=$action, effective_delta=$effective_delta)"

        before=$(fetch_service_metrics "$svc")
        echo "  Before: cpu,rps,err,lat = $before"

        if [ "$r_new" -ne "$r_old" ]; then
            scale_service "$svc" "$r_new"
        fi

        sleep "$SETTLE_AFTER"

        after=$(fetch_service_metrics "$svc")
        echo "  After:  cpu,rps,err,lat = $after"

        ts=$(date +%s)

        echo "${ts},${load},${svc},${r_old},${r_new},${action},${effective_delta},${before},${after}" >> "$OUTFILE"
    done

    stop_locust
    sleep 10
done

echo ""
echo "======================================================"
echo " DONE — Action-effect pairs saved to: $OUTFILE"
echo " Total trials: $TRIAL_NUM"
echo "======================================================"

echo ""
echo "Sample count by load:"
cut -d, -f2 "$OUTFILE" | tail -n +2 | sort | uniq -c || true

echo ""
echo "Sample count by service:"
cut -d, -f3 "$OUTFILE" | tail -n +2 | sort | uniq -c || true

echo ""
echo "Sample count by effective_delta:"
cut -d, -f7 "$OUTFILE" | tail -n +2 | sort | uniq -c || true