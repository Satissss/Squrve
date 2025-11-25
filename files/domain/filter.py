import json

with open("finance.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(list(data.keys()))