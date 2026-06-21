#!/bin/bash
#!/bin/bash

# ============================================================
# Diverse Long-Running Load Generator
# ============================================================

WORKER_IP="192.168.120.185"

TARGET="http://$WORKER_IP:30080"
PROMETHEUS="http://$WORKER_IP:30090"

LOCUSTFILE="$(cd $(dirname $0) && pwd)/locustfile.py"

# ============================================================
# Helpers
# ============================================================

stop_locust() {
    pkill -f "locust" 2>/dev/null
    sleep 3
}

run_locust() {
    local users=$1
    local spawn=$2
    local runtime=$3
    local phase=$4

    echo ""
    echo "======================================================"
    echo "[$(date +%H:%M:%S)] $phase"
    echo "Users      : $users"
    echo "Spawn Rate : $spawn"
    echo "Duration   : ${runtime}s"
    echo "======================================================"

    locust -f "$LOCUSTFILE" \
        --host "$TARGET" \
        --headless \
        --users "$users" \
        --spawn-rate "$spawn" \
        --run-time "${runtime}s" \
        --loglevel WARNING
}

# ============================================================
# Init
# ============================================================

echo ""
echo "======================================================"
echo " Online Boutique Diverse Workload"
echo "======================================================"
echo "Target      : $TARGET"
echo "Prometheus  : $PROMETHEUS"
echo "Locustfile  : $LOCUSTFILE"
echo "======================================================"

if [ ! -f "$LOCUSTFILE" ]; then
    echo "ERROR: locustfile.py not found!"
    exit 1
fi

# ============================================================
# PHASE 0 — Warmup (5 min)
# ============================================================

stop_locust
run_locust 5 1 300 "PHASE 0 — WARMUP"

sleep 10

# ============================================================
# PHASE 1 — Low Stable Traffic (20 min)
# ============================================================

stop_locust
run_locust 15 2 1200 "PHASE 1 — LOW STABLE TRAFFIC"

sleep 10

# ============================================================
# PHASE 2 — Medium Stable Traffic (20 min)
# ============================================================

stop_locust
run_locust 40 4 1200 "PHASE 2 — MEDIUM STABLE TRAFFIC"

sleep 10

# ============================================================
# PHASE 3 — Medium-High Traffic (20 min)
# ============================================================

stop_locust
run_locust 70 5 1200 "PHASE 3 — MEDIUM-HIGH TRAFFIC"

sleep 10

# ============================================================
# PHASE 4 — Controlled Ramp Up (20 min)
# ============================================================

for users in 80 100 120 140 160; do
    stop_locust

    run_locust \
        "$users" \
        8 \
        240 \
        "PHASE 4 — RAMP USERS=$users"

    sleep 10
done

# ============================================================
# PHASE 5 — High Steady State (20 min)
# ============================================================

stop_locust
run_locust 160 10 1200 "PHASE 5 — HIGH STEADY LOAD"

sleep 10

# ============================================================
# PHASE 6 — Burst Windows (18 min)
# ============================================================

for users in 200 60 220 70 180 50; do

    stop_locust

    if [ "$users" -ge 180 ]; then
        run_locust \
            "$users" \
            15 \
            180 \
            "PHASE 6 — BURST USERS=$users"
    else
        run_locust \
            "$users" \
            5 \
            180 \
            "PHASE 6 — RECOVERY USERS=$users"
    fi

    sleep 10
done

# ============================================================
# PHASE 7 — Oscillation (24 min)
# ============================================================

OSCILLATION=(30 60 90 120 90 60 30 80)

for users in "${OSCILLATION[@]}"; do
    stop_locust

    run_locust \
        "$users" \
        6 \
        180 \
        "PHASE 7 — OSCILLATION USERS=$users"

    sleep 10
done

# ============================================================
# PHASE 8 — Cooldown (15 min)
# ============================================================

stop_locust
run_locust 10 2 900 "PHASE 8 — COOLDOWN"

# ============================================================
# END
# ============================================================

stop_locust

echo ""
echo "======================================================"
echo " ALL PHASES COMPLETE"
echo "======================================================"

# ============================================================
# METRIC COLLECTION
# ============================================================

# Total runtime ≈ 9720s (~2h 42m)
DURATION=9800

echo ""
echo "Collecting Prometheus metrics..."

python3 "$(dirname $0)/collect_and_preprocess.py" \
    --prometheus "$PROMETHEUS" \
    --duration $DURATION \
    --step 15 \
    --outdir ./data

echo ""
echo "Done."
echo "Output:"
echo "  ./data/nodes_data.csv"
echo "  ./data/edges_data.csv"
