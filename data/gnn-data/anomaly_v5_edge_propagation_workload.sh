#!/bin/bash

WORKER_IP="192.168.120.185"
TARGET="http://$WORKER_IP:30080"
PROMETHEUS="http://$WORKER_IP:30090"
LOCUSTFILE="./locustfile.py"
OUTDIR="./dataset-v5/data_deep_anomaly"
TIMELINE="./anomaly_timeline_v5.csv"

mkdir -p "$OUTDIR"
echo "start_ts,end_ts,label,anomaly_type,affected_service,affected_edge" > "$TIMELINE"

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

record_anomaly() {
    local start_ts=$1
    local end_ts=$2
    local anomaly_type=$3
    local service=$4
    local edge=$5

    echo "$start_ts,$end_ts,1,$anomaly_type,$service,$edge" >> "$TIMELINE"
}

cleanup_faults() {
    kubectl delete virtualservice productcatalogservice-v5-fault -n online-boutique --ignore-not-found
    kubectl delete virtualservice cartservice-v5-fault -n online-boutique --ignore-not-found
    kubectl delete virtualservice paymentservice-v5-fault -n online-boutique --ignore-not-found
    kubectl delete virtualservice checkoutservice-v5-fault -n online-boutique --ignore-not-found
    kubectl delete virtualservice recommendationservice-v5-fault -n online-boutique --ignore-not-found
}

cleanup_faults
stop_locust

START_ALL=$(date +%s)

echo "======================================================"
echo " DATASET-V5 EDGE PROPAGATION ANOMALY WORKLOAD"
echo "======================================================"

# ============================================================
# PHASE 0 — BASELINE
# ============================================================

run_load 60 6 600 "PHASE 0 — BASELINE BEFORE ANOMALY"
sleep 10

# ============================================================
# PHASE 1 — PRODUCTCATALOG LATENCY ONLY
# frontend -> productcatalogservice
# recommendationservice -> productcatalogservice
# ============================================================

cat > productcatalogservice-v5-fault.yaml <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: productcatalogservice-v5-fault
  namespace: online-boutique
spec:
  hosts:
  - productcatalogservice
  http:
  - fault:
      delay:
        percentage:
          value: 40
        fixedDelay: 900ms
    route:
    - destination:
        host: productcatalogservice
EOF

A_START=$(date +%s)
kubectl apply -f productcatalogservice-v5-fault.yaml
run_load 90 8 600 "PHASE 1 — LATENCY ONLY productcatalogservice"
A_END=$(date +%s)

record_anomaly \
  "$A_START" \
  "$A_END" \
  "latency_only_productcatalog_read_path" \
  "productcatalogservice" \
  "frontend->productcatalogservice|recommendationservice->productcatalogservice"

kubectl delete -f productcatalogservice-v5-fault.yaml
sleep 10

run_load 50 5 420 "RECOVERY PRODUCTCATALOG"
sleep 10

# ============================================================
# PHASE 2 — CARTSERVICE ERROR ONLY
# frontend -> cartservice
# checkoutservice -> cartservice
# ============================================================

cat > cartservice-v5-fault.yaml <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: cartservice-v5-fault
  namespace: online-boutique
spec:
  hosts:
  - cartservice
  http:
  - fault:
      abort:
        percentage:
          value: 7
        httpStatus: 500
    route:
    - destination:
        host: cartservice
EOF

A_START=$(date +%s)
kubectl apply -f cartservice-v5-fault.yaml
run_load 90 8 600 "PHASE 2 — ERROR ONLY cartservice"
A_END=$(date +%s)

record_anomaly \
  "$A_START" \
  "$A_END" \
  "error_only_cartservice_write_path" \
  "cartservice" \
  "frontend->cartservice|checkoutservice->cartservice"

kubectl delete -f cartservice-v5-fault.yaml
sleep 10

run_load 50 5 420 "RECOVERY CARTSERVICE ERROR ONLY"
sleep 10

# ============================================================
# PHASE 3 — CARTSERVICE LATENCY + ERROR
# medium degradation
# ============================================================

cat > cartservice-v5-fault.yaml <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: cartservice-v5-fault
  namespace: online-boutique
spec:
  hosts:
  - cartservice
  http:
  - fault:
      delay:
        percentage:
          value: 45
        fixedDelay: 1200ms
      abort:
        percentage:
          value: 8
        httpStatus: 500
    route:
    - destination:
        host: cartservice
EOF

A_START=$(date +%s)
kubectl apply -f cartservice-v5-fault.yaml
run_load 110 10 600 "PHASE 3 — MEDIUM cartservice latency + error"
A_END=$(date +%s)

record_anomaly \
  "$A_START" \
  "$A_END" \
  "medium_cartservice_latency_error" \
  "cartservice" \
  "frontend->cartservice|checkoutservice->cartservice"

kubectl delete -f cartservice-v5-fault.yaml
sleep 10

run_load 50 5 420 "RECOVERY CARTSERVICE MIXED"
sleep 10

# ============================================================
# PHASE 4 — PAYMENT CHECKOUT CHAIN
# checkoutservice -> paymentservice
# ============================================================

cat > paymentservice-v5-fault.yaml <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: paymentservice-v5-fault
  namespace: online-boutique
spec:
  hosts:
  - paymentservice
  http:
  - fault:
      delay:
        percentage:
          value: 50
        fixedDelay: 1500ms
      abort:
        percentage:
          value: 10
        httpStatus: 500
    route:
    - destination:
        host: paymentservice
EOF

A_START=$(date +%s)
kubectl apply -f paymentservice-v5-fault.yaml
run_load 110 10 600 "PHASE 4 — PAYMENT checkout-chain degradation"
A_END=$(date +%s)

record_anomaly \
  "$A_START" \
  "$A_END" \
  "paymentservice_checkout_chain_degradation" \
  "paymentservice" \
  "checkoutservice->paymentservice"

kubectl delete -f paymentservice-v5-fault.yaml
sleep 10

run_load 50 5 420 "RECOVERY PAYMENT"
sleep 10

# ============================================================
# PHASE 5 — CHECKOUT UPSTREAM CASCADE
# frontend -> checkoutservice
# ============================================================

cat > checkoutservice-v5-fault.yaml <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: checkoutservice-v5-fault
  namespace: online-boutique
spec:
  hosts:
  - checkoutservice
  http:
  - fault:
      delay:
        percentage:
          value: 35
        fixedDelay: 1000ms
      abort:
        percentage:
          value: 5
        httpStatus: 500
    route:
    - destination:
        host: checkoutservice
EOF

A_START=$(date +%s)
kubectl apply -f checkoutservice-v5-fault.yaml
run_load 90 8 480 "PHASE 5 — CHECKOUT upstream mild cascade"
A_END=$(date +%s)

record_anomaly \
  "$A_START" \
  "$A_END" \
  "checkoutservice_upstream_mild_cascade" \
  "checkoutservice" \
  "frontend->checkoutservice"

kubectl delete -f checkoutservice-v5-fault.yaml
sleep 10

run_load 50 5 420 "RECOVERY CHECKOUT"
sleep 10

# ============================================================
# PHASE 6 — RECOMMENDATION LATENCY
# frontend -> recommendationservice
# recommendationservice -> productcatalogservice
# ============================================================

cat > recommendationservice-v5-fault.yaml <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: recommendationservice-v5-fault
  namespace: online-boutique
spec:
  hosts:
  - recommendationservice
  http:
  - fault:
      delay:
        percentage:
          value: 40
        fixedDelay: 1000ms
    route:
    - destination:
        host: recommendationservice
EOF

A_START=$(date +%s)
kubectl apply -f recommendationservice-v5-fault.yaml
run_load 80 8 480 "PHASE 6 — RECOMMENDATION latency degradation"
A_END=$(date +%s)

record_anomaly \
  "$A_START" \
  "$A_END" \
  "recommendationservice_latency_degradation" \
  "recommendationservice" \
  "frontend->recommendationservice|recommendationservice->productcatalogservice"

kubectl delete -f recommendationservice-v5-fault.yaml
sleep 10

run_load 50 5 420 "RECOVERY RECOMMENDATION"
sleep 10

# ============================================================
# PHASE 7 — NO FAULT OSCILLATION
# hard normal-like traffic inside anomaly session
# ============================================================

run_load 70 7 300 "PHASE 7A — NO FAULT OSCILLATION 70 USERS"
sleep 10

run_load 120 10 300 "PHASE 7B — NO FAULT OSCILLATION 120 USERS"
sleep 10

run_load 60 6 300 "PHASE 7C — NO FAULT OSCILLATION 60 USERS"
sleep 10

# ============================================================
# PHASE 8 — FINAL RECOVERY
# ============================================================

run_load 40 4 600 "PHASE 8 — FINAL RECOVERY"

END_ALL=$(date +%s)
DURATION=$((END_ALL - START_ALL + 240))

cleanup_faults
stop_locust

echo ""
echo "======================================================"
echo "Collecting anomaly metrics"
echo "Duration=$DURATION seconds"
echo "======================================================"

python3 collect_and_preprocess.py \
    --prometheus "$PROMETHEUS" \
    --duration "$DURATION" \
    --step 15 \
    --outdir "$OUTDIR"

cp "$TIMELINE" "$OUTDIR/anomaly_timeline.csv"

echo ""
echo "DONE: $OUTDIR"
echo "Files:"
echo "  $OUTDIR/nodes_data.csv"
echo "  $OUTDIR/edges_data.csv"
echo "  $OUTDIR/anomaly_timeline.csv"
