#!/bin/bash

WORKER_IP="192.168.120.185"
TARGET="http://$WORKER_IP:30080"
PROMETHEUS="http://$WORKER_IP:30090"
LOCUSTFILE="./locustfile.py"
OUTDIR="./data_deep_anomaly"
TIMELINE="./anomaly_timeline.csv"

mkdir -p "$OUTDIR"
echo "start_ts,end_ts,label,anomaly_type,affected_service,affected_edge" > "$TIMELINE"

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
    kubectl delete virtualservice productcatalogservice-delay -n online-boutique --ignore-not-found
    kubectl delete virtualservice cartservice-delay-abort -n online-boutique --ignore-not-found
    kubectl delete virtualservice paymentservice-delay-abort -n online-boutique --ignore-not-found
}

cleanup_faults

START_ALL=$(date +%s)

echo "=== PHASE 1: NORMAL BASELINE 10 MIN ==="
run_load 50 5 600 "NORMAL BASELINE"

echo "=== PHASE 2: PRODUCTCATALOG BOTTLENECK ==="
cat > productcatalog-delay.yaml <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: productcatalogservice-delay
  namespace: online-boutique
spec:
  hosts:
  - productcatalogservice
  http:
  - fault:
      delay:
        percentage:
          value: 50
        fixedDelay: 1500ms
    route:
    - destination:
        host: productcatalogservice
EOF

A_START=$(date +%s)
kubectl apply -f productcatalog-delay.yaml
run_load 120 10 480 "ANOMALY productcatalogservice delay"
A_END=$(date +%s)
record_anomaly "$A_START" "$A_END" "productcatalog_latency_bottleneck" "productcatalogservice" "frontend->productcatalogservice|recommendationservice->productcatalogservice"
kubectl delete -f productcatalog-delay.yaml

echo "=== PHASE 3: RECOVERY 5 MIN ==="
run_load 50 5 300 "RECOVERY AFTER PRODUCTCATALOG"

echo "=== PHASE 4: CARTSERVICE BOTTLENECK + ERROR ==="
cat > cartservice-delay-abort.yaml <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: cartservice-delay-abort
  namespace: online-boutique
spec:
  hosts:
  - cartservice
  http:
  - fault:
      delay:
        percentage:
          value: 50
        fixedDelay: 2000ms
      abort:
        percentage:
          value: 15
        httpStatus: 500
    route:
    - destination:
        host: cartservice
EOF

A_START=$(date +%s)
kubectl apply -f cartservice-delay-abort.yaml
run_load 130 12 480 "ANOMALY cartservice delay + 500"
A_END=$(date +%s)
record_anomaly "$A_START" "$A_END" "cartservice_latency_error_bottleneck" "cartservice" "frontend->cartservice|checkoutservice->cartservice"
kubectl delete -f cartservice-delay-abort.yaml

echo "=== PHASE 5: RECOVERY 5 MIN ==="
run_load 50 5 300 "RECOVERY AFTER CARTSERVICE"

echo "=== PHASE 6: PAYMENTSERVICE CHECKOUT CHAIN ERROR ==="
cat > paymentservice-delay-abort.yaml <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: paymentservice-delay-abort
  namespace: online-boutique
spec:
  hosts:
  - paymentservice
  http:
  - fault:
      delay:
        percentage:
          value: 60
        fixedDelay: 2500ms
      abort:
        percentage:
          value: 20
        httpStatus: 500
    route:
    - destination:
        host: paymentservice
EOF

A_START=$(date +%s)
kubectl apply -f paymentservice-delay-abort.yaml
run_load 140 12 480 "ANOMALY paymentservice delay + 500"
A_END=$(date +%s)
record_anomaly "$A_START" "$A_END" "paymentservice_checkout_chain_error" "paymentservice" "checkoutservice->paymentservice"
kubectl delete -f paymentservice-delay-abort.yaml

echo "=== PHASE 7: FINAL RECOVERY 10 MIN ==="
run_load 40 4 600 "FINAL RECOVERY"

END_ALL=$(date +%s)
DURATION=$((END_ALL - START_ALL + 120))

cleanup_faults

echo "=== COLLECTING METRICS ==="
echo "Duration=$DURATION seconds"

python3 collect_and_preprocess.py \
    --prometheus "$PROMETHEUS" \
    --duration "$DURATION" \
    --step 15 \
    --outdir "$OUTDIR"

cp "$TIMELINE" "$OUTDIR/anomaly_timeline.csv"

echo "=== DONE ==="
echo "Output: $OUTDIR"
