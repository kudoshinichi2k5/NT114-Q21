#!/bin/bash

WORKER_IP="192.168.120.185"
TARGET="http://$WORKER_IP:30080"
PROMETHEUS="http://$WORKER_IP:30090"
LOCUSTFILE="./locustfile.py"
OUTDIR="./dataset-v5/data_final_cascade_anomaly"
TIMELINE="./anomaly_timeline_v5_final_cascade.csv"

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
    echo "$1,$2,1,$3,$4,$5" >> "$TIMELINE"
}

cleanup_faults() {
    kubectl delete virtualservice productcatalogservice-final-cascade -n online-boutique --ignore-not-found
    kubectl delete virtualservice checkoutservice-final-cascade -n online-boutique --ignore-not-found
    kubectl delete virtualservice paymentservice-final-cascade -n online-boutique --ignore-not-found
}

cleanup_faults
stop_locust

START_ALL=$(date +%s)

run_load 60 6 480 "BASELINE BEFORE FINAL CASCADE"
sleep 10

cat > productcatalogservice-final-cascade.yaml <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: productcatalogservice-final-cascade
  namespace: online-boutique
spec:
  hosts:
  - productcatalogservice
  http:
  - fault:
      delay:
        percentage:
          value: 55
        fixedDelay: 1600ms
      abort:
        percentage:
          value: 8
        httpStatus: 500
    route:
    - destination:
        host: productcatalogservice
EOF

A_START=$(date +%s)
kubectl apply -f productcatalogservice-final-cascade.yaml
run_load 120 12 600 "FINAL CASCADE 1 — productcatalog read-path bottleneck"
A_END=$(date +%s)
record_anomaly "$A_START" "$A_END" "final_productcatalog_read_path_cascade" "productcatalogservice" "frontend->productcatalogservice|recommendationservice->productcatalogservice"
kubectl delete -f productcatalogservice-final-cascade.yaml
sleep 10

run_load 50 5 420 "RECOVERY AFTER PRODUCTCATALOG"
sleep 10

cat > checkoutservice-final-cascade.yaml <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: checkoutservice-final-cascade
  namespace: online-boutique
spec:
  hosts:
  - checkoutservice
  http:
  - fault:
      delay:
        percentage:
          value: 50
        fixedDelay: 1500ms
      abort:
        percentage:
          value: 8
        httpStatus: 500
    route:
    - destination:
        host: checkoutservice
EOF

cat > paymentservice-final-cascade.yaml <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: paymentservice-final-cascade
  namespace: online-boutique
spec:
  hosts:
  - paymentservice
  http:
  - fault:
      delay:
        percentage:
          value: 45
        fixedDelay: 1800ms
      abort:
        percentage:
          value: 10
        httpStatus: 500
    route:
    - destination:
        host: paymentservice
EOF

A_START=$(date +%s)
kubectl apply -f checkoutservice-final-cascade.yaml
kubectl apply -f paymentservice-final-cascade.yaml
run_load 120 12 600 "FINAL CASCADE 2 — checkout + payment coupled degradation"
A_END=$(date +%s)
record_anomaly "$A_START" "$A_END" "final_checkout_payment_coupled_cascade" "checkoutservice|paymentservice" "frontend->checkoutservice|checkoutservice->paymentservice"
kubectl delete -f checkoutservice-final-cascade.yaml
kubectl delete -f paymentservice-final-cascade.yaml
sleep 10

run_load 70 7 300 "POST-CASCADE RECOVERY 70 USERS"
sleep 10
run_load 40 4 600 "FINAL RECOVERY"

END_ALL=$(date +%s)
DURATION=$((END_ALL - START_ALL + 240))

cleanup_faults
stop_locust

python3 collect_and_preprocess.py \
    --prometheus "$PROMETHEUS" \
    --duration "$DURATION" \
    --step 15 \
    --outdir "$OUTDIR"

cp "$TIMELINE" "$OUTDIR/anomaly_timeline.csv"

echo "DONE: $OUTDIR"
