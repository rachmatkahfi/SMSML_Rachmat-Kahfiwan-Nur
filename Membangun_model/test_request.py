import requests

url = "http://127.0.0.1:8080/predict"
payload = {"features": [1, 2, 3, 4, 5, 6, 7, 8]}

response = requests.post(url, json=payload)
print(response.json())
