#!/bin/bash

WORKER_IP="192.168.120.185"
TARGET="http://$WORKER_IP:30080"
PROMETHEUS="http://$WORKER_IP:30090"
LOCUSTFILE="./locustfile.py"
OUTDIR="./dataset-v4/data_normal"

mkdir -p "$OUTDIR"

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

START_TS=$(date +%s)

run_load 20 2 600 "NORMAL LOW 10 MIN"
run_load 60 5 600 "NORMAL MEDIUM 10 MIN"
run_load 100 8 600 "NORMAL HIGH 10 MIN"
run_load 40 4 600 "NORMAL RECOVERY 10 MIN"

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS + 120))

echo "Collecting normal metrics, duration=$DURATION"

python3 collect_and_preprocess.py \
    --prometheus "$PROMETHEUS" \
    --duration "$DURATION" \
    --step 15 \
    --outdir "$OUTDIR"

echo "DONE: $OUTDIR"
