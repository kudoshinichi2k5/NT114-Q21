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

echo "======================================================"
echo " ANOMALY V5 FULL WORKLOAD"
echo " Target: $TARGET"
echo " Output: $OUTDIR"
echo "======================================================"

START_ALL=$(date +%s)

run_load 60 6 600 "BASELINE BEFORE ANOMALY 10 MIN"
sleep 10

echo "=== PHASE 1: LATENCY-ONLY PRODUCTCATALOG READ PATH ==="

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
run_load 90 8 600 "LATENCY-ONLY productcatalogservice"
A_END=$(date +%s)
record_anomaly "$A_START" "$A_END" "latency_only_productcatalog_read_path" "productcatalogservice" "frontend->productcatalogservice|recommendationservice->productcatalogservice"
kubectl delete -f productcatalogservice-v5-fault.yaml
sleep 10

run_load 50 5 420 "RECOVERY PRODUCTCATALOG"
sleep 10

echo "=== PHASE 2: ERROR-ONLY CARTSERVICE WRITE PATH ==="

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
run_load 90 8 600 "ERROR-ONLY cartservice"
A_END=$(date +%s)
record_anomaly "$A_START" "$A_END" "error_only_cartservice_write_path" "cartservice" "frontend->cartservice|checkoutservice->cartservice"
kubectl delete -f cartservice-v5-fault.yaml
sleep 10

run_load 50 5 420 "RECOVERY CARTSERVICE"
sleep 10

echo "=== PHASE 3: MEDIUM CARTSERVICE LATENCY + ERROR ==="

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
run_load 110 10 600 "MEDIUM cartservice latency + error"
A_END=$(date +%s)
record_anomaly "$A_START" "$A_END" "medium_cartservice_latency_error" "cartservice" "frontend->cartservice|checkoutservice->cartservice"
kubectl delete -f cartservice-v5-fault.yaml
sleep 10

run_load 50 5 420 "RECOVERY CARTSERVICE MEDIUM"
sleep 10

echo "=== PHASE 4: PAYMENT CHECKOUT CHAIN DEGRADATION ==="

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
run_load 110 10 600 "PAYMENT checkout-chain degradation"
A_END=$(date +%s)
record_anomaly "$A_START" "$A_END" "paymentservice_checkout_chain_degradation" "paymentservice" "checkoutservice->paymentservice"
kubectl delete -f paymentservice-v5-fault.yaml
sleep 10

run_load 50 5 420 "RECOVERY PAYMENT"
sleep 10

echo "=== PHASE 5: CHECKOUT UPSTREAM MILD CASCADE ==="

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
run_load 90 8 480 "CHECKOUT upstream mild cascade"
A_END=$(date +%s)
record_anomaly "$A_START" "$A_END" "checkoutservice_upstream_mild_cascade" "checkoutservice" "frontend->checkoutservice"
kubectl delete -f checkoutservice-v5-fault.yaml
sleep 10

run_load 50 5 420 "RECOVERY CHECKOUT"
sleep 10

echo "=== PHASE 6: RECOMMENDATION PATH LATENCY ==="

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
run_load 80 8 480 "RECOMMENDATION latency degradation"
A_END=$(date +%s)
record_anomaly "$A_START" "$A_END" "recommendationservice_latency_degradation" "recommendationservice" "frontend->recommendationservice"
kubectl delete -f recommendationservice-v5-fault.yaml
sleep 10

run_load 50 5 420 "RECOVERY RECOMMENDATION"
sleep 10

echo "=== PHASE 7: MIXED TRAFFIC OSCILLATION WITHOUT FAULT ==="

run_load 70 7 300 "OSCILLATION NO FAULT 70 USERS"
sleep 10
run_load 120 10 300 "OSCILLATION NO FAULT 120 USERS"
sleep 10
run_load 60 6 300 "OSCILLATION NO FAULT 60 USERS"
sleep 10

run_load 40 4 600 "FINAL RECOVERY 10 MIN"

END_ALL=$(date +%s)
DURATION=$((END_ALL - START_ALL + 240))

cleanup_faults
stop_locust

echo "Collecting anomaly metrics, duration=$DURATION"

python3 collect_and_preprocess.py \
    --prometheus "$PROMETHEUS" \
    --duration "$DURATION" \
    --step 15 \
    --outdir "$OUTDIR"

cp "$TIMELINE" "$OUTDIR/anomaly_timeline.csv"

echo "DONE: $OUTDIR"
