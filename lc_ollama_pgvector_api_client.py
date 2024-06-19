import requests
import os , sys

payload = { "question" :  sys.argv[1] }

response = requests.post(
    "http://localhost:8000/ragQuery/invoke",
    json={"input": sys.argv[1] }
)

print(response.json())
