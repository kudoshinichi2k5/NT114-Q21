# locustfile.py
from locust import HttpUser, TaskSet, task, between
import random

PRODUCTS = [
    "OLJCESPC7Z", "66VCHSJNUP", "1YMWWN1N4O",
    "L9ECAV7KIM", "2ZYFJ3GM2N", "0PUK6V6EV0",
    "LS4PSXUNUM", "9SIQT8TOJO", "6E92ZMYYFZ",
]

class ShoppingBehavior(TaskSet):
    @task(5)
    def browse_homepage(self):
        self.client.get("/", name="[GET] homepage")

    @task(4)
    def view_product(self):
        pid = random.choice(PRODUCTS)
        self.client.get(f"/product/{pid}", name="[GET] product")

    @task(3)
    def add_to_cart(self):
        pid = random.choice(PRODUCTS)
        self.client.post("/cart", json={
            "product_id": pid,
            "quantity": random.randint(1, 3),
        }, name="[POST] add-to-cart")

    @task(2)
    def view_cart(self):
        self.client.get("/cart", name="[GET] cart")

    @task(1)
    def checkout(self):
        self.client.post("/cart/checkout", data={
            "email": "test@example.com",
            "street_address": "123 Main St",
            "zip_code": "10001",
            "city": "New York",
            "state": "NY",
            "country": "US",
            "credit_card_number": "4432801561520454",
            "credit_card_expiration_month": "1",
            "credit_card_expiration_year": "2039",
            "credit_card_cvv": "672",
        }, name="[POST] checkout")

    @task(2)
    def currency(self):
        self.client.post("/setCurrency", data={
            "currency_code": random.choice(["USD", "EUR", "JPY", "GBP"])
        }, name="[POST] set-currency")

class OnlineBoutiqueUser(HttpUser):
    tasks = [ShoppingBehavior]
    wait_time = between(1, 5)
