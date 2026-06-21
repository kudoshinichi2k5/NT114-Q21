#!/bin/bash

WORKER_IP="192.168.120.185"
TARGET="http://$WORKER_IP:30080"
PROMETHEUS="http://$WORKER_IP:30090"
LOCUSTFILE="./locustfile.py"
OUTDIR="./dataset-v5/data_normal"

mkdir -p "$OUTDIR"

stop_locust() {
    pkill -f "locust" 2>/dev/null
    sleep 3
}

run_load() {
    local users=$1
    local spawn=$2
    local runtime=$3
    local phase=$4

    echo ""
    echo "======================================================"
    echo "[$(date +%H:%M:%S)] $phase"
    echo "Users=$users | Spawn=$spawn | Runtime=${runtime}s"
    echo "======================================================"

    locust -f "$LOCUSTFILE" \
        --host "$TARGET" \
        --headless \
        --users "$users" \
        --spawn-rate "$spawn" \
        --run-time "${runtime}s" \
        --loglevel WARNING
}

echo "======================================================"
echo " NORMAL V5 FULL WORKLOAD"
echo " Target: $TARGET"
echo " Output: $OUTDIR"
echo "======================================================"

START_TS=$(date +%s)

stop_locust

run_load 10 2 600 "NORMAL WARMUP 10 MIN"
sleep 10

run_load 25 3 900 "NORMAL LOW STABLE 15 MIN"
sleep 10

run_load 60 6 900 "NORMAL MEDIUM STABLE 15 MIN"
sleep 10

run_load 100 8 900 "NORMAL HIGH STABLE 15 MIN"
sleep 10

for users in 40 80 120 80 50; do
    run_load "$users" 8 300 "NORMAL OSCILLATION USERS=$users"
    sleep 10
done

run_load 140 10 420 "NORMAL SHORT PEAK 7 MIN"
sleep 10

run_load 70 6 600 "NORMAL RECOVERY MEDIUM 10 MIN"
sleep 10

run_load 30 3 600 "NORMAL COOLDOWN 10 MIN"

stop_locust

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS + 240))

echo ""
echo "Collecting normal metrics, duration=$DURATION"

python3 collect_and_preprocess.py \
    --prometheus "$PROMETHEUS" \
    --duration "$DURATION" \
    --step 15 \
    --outdir "$OUTDIR"

echo "DONE: $OUTDIR"
