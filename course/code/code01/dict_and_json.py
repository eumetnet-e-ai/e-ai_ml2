import json

cfg = {"dt": 0.5, "n": 20, "model": "toy"}
s = json.dumps(cfg, indent=2)
print(s)

cfg2 = json.loads(s)
print("dt =", cfg2["dt"])
