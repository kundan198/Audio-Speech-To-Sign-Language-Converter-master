import requests

url = "http://127.0.0.1:8001/simplify"
payload = {
    "text": "I am going to the college to learn computer language.",
    "gemini_key": "AIzaSyBaOuMgoe3mudTuiFPneVNKyUqz-PnTt8s"
}

try:
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
