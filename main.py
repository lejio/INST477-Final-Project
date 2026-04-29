import requests

r = requests.get("https://apewisdom.io/api/v1.0/filter/stocks")

if r.status_code == 200:
    print(r.json())

