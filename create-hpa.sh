SERVICES=("frontend" "checkoutservice" "currencyservice" "cartservice" "productcatalogservice" "recommendationservice" "shippingservice" "adservice" "emailservice" "paymentservice")

for svc in "${SERVICES[@]}"; do
  kubectl autoscale deployment $svc -n online-boutique \
    --cpu-percent=70 \
    --min=1 \
    --max=5
  echo "Đã tạo HPA cho $svc"
done
