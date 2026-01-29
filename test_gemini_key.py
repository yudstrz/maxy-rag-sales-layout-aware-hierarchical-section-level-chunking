import requests
import os

API_KEY = "AIzaSyBi55vB-zhismRhNibC0iEjD4-5kUmkuoo"
MODEL = "gemini-flash-latest"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

def test_gemini():
    url = f"{BASE_URL}/models/{MODEL}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": "Hello, are you working?"}]}]
    }
    
    print(f"Testing Gemini API...")
    print(f"Model: {MODEL}")
    print(f"URL: {url.replace(API_KEY, 'HIDDEN')}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_gemini()
