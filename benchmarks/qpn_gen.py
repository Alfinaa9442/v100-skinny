"""Capture the four ladder-cell outputs (greedy) for byte diffs.
Usage: qpn_gen.py <tag>  -> writes lad_<cell>_<tag>.txt files."""
import json
import sys
import urllib.request

TAG = sys.argv[1]
BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.6-27b-nvfp4-skinny"

EXTRACT_PASSAGE = ("The lighthouse keeper checked the lamp at dusk. Salt wind "
                   "pressed against the tower glass. Ships passed far out. ") * 30
CELLS = [
    ("math", "A train leaves station A at 60 km/h. Two hours later a "
             "second train leaves the same station at 90 km/h on a "
             "parallel track. How far from the station does the second "
             "train catch the first? Work through this carefully.",
     True, 2048),
    ("json", "Generate a JSON array of 40 user records with fields id "
             "(sequential integer), name, email, signup_date (ISO 8601), "
             "and plan (one of free/pro/enterprise). Output only the JSON.",
     False, 1024),
    ("csv", "Produce a CSV table with 50 rows and columns product_id, "
            "name, category, price, stock. Realistic hardware-store "
            "products. Output only the CSV.",
     False, 1024),
    ("extract", "Repeat the following passage exactly:\n\n" + EXTRACT_PASSAGE,
     False, 512),
]

for cell, prompt, thinking, mt in CELLS:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_completion_tokens": mt,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    msg = d["choices"][0]["message"]
    text = (msg.get("reasoning") or "") + "\x1d" + (msg.get("content") or "")
    with open(f"/home/user/flatness-run/lad_{cell}_{TAG}.txt", "w") as f:
        f.write(text)
    print(f"saved lad_{cell}_{TAG}.txt "
          f"({d['usage']['completion_tokens']} tok)")
