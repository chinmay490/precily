import requests

r = requests.post('http://127.0.0.1:5000/score', json={
    "text1": "Please bro work",
    "text2": "Let it be for now"
})
print(f"Status Code: {r.status_code}, Response: {r.json()}")