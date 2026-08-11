"""Per-k workload probe: acceptance (incl. per-position) + wall tok/s
on the math/code fixture cells. Run once per booted k; k passed as argv.
Appends CSV rows to results/k_sweep_matrix.csv (SKINNY_OUT_DIR overrides).
"""
import json
import sys
import time
import urllib.request

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import kernel_src, out_csv, fixtures_dir  # noqa: E402

BASE = "http://127.0.0.1:8000"
MODEL = _os.environ.get("PROBE_MODEL", "qwen3.6-27b-nvfp4-skinny")
K = int(sys.argv[1])
CELL_SET = sys.argv[2] if len(sys.argv) > 2 else "mathcode"
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
OUT = out_csv("k_sweep_matrix.csv")

MATH_PROMPT = ("A train leaves station A at 60 km/h. Two hours later a "
               "second train leaves the same station at 90 km/h on a "
               "parallel track. How far from the station does the second "
               "train catch the first? Work through this carefully.")
CODE_PROMPT = ("Write a complete Python implementation of a red-black "
               "tree with insert, delete, and search, with docstrings.")
PROSE_PROMPT = ("Write a very long detailed story about a lighthouse "
                "keeper. Keep going, do not stop.")

CELL_SETS = {
    "mathcode": [
        ("math-think-2048", MATH_PROMPT, True, 2048),
        ("math-nothink-1024", MATH_PROMPT, False, 1024),
        ("code-nothink-1024", CODE_PROMPT, False, 1024),
        ("code-think-2048", CODE_PROMPT, True, 2048),
    ],
    "prose": [
        ("prose-think-1024", PROSE_PROMPT, True, 1024),
        ("prose-nothink-1024", PROSE_PROMPT, False, 1024),
    ],
    "both": [],
    "codeboiler": [
        ("boiler-crud-think-2048",
         "Write a complete FastAPI application with CRUD endpoints "
         "(list, get, create, update, delete) for five resources: users, "
         "products, orders, invoices, and shipments. Each follows the "
         "same pattern with Pydantic models.", True, 2048),
        ("boiler-crud-nothink-1024",
         "Write a complete FastAPI application with CRUD endpoints "
         "(list, get, create, update, delete) for five resources: users, "
         "products, orders, invoices, and shipments. Each follows the "
         "same pattern with Pydantic models.", False, 1024),
        ("boiler-sql-nothink-1024",
         "Write SQL DDL: CREATE TABLE statements for 15 tables of an "
         "e-commerce schema, each with id, created_at, updated_at, and "
         "4-6 domain columns with sensible types and foreign keys.",
         False, 1024),
    ],
    "structured": [
        ("struct-json-1024",
         "Generate a JSON array of 40 user records with fields id "
         "(sequential integer), name, email, signup_date (ISO 8601), and "
         "plan (one of free/pro/enterprise). Output only the JSON.",
         False, 1024),
        ("struct-csv-1024",
         "Produce a CSV table with 50 rows and columns product_id, name, "
         "category, price, stock. Realistic hardware-store products. "
         "Output only the CSV.",
         False, 1024),
    ],
    "gate": [
        ("prose-think-2048", PROSE_PROMPT, True, 2048),
        ("math-think-2048", MATH_PROMPT, True, 2048),
        ("prose-nothink-1024", PROSE_PROMPT, False, 1024),
    ],
}
CELL_SETS["both"] = CELL_SETS["prose"] + CELL_SETS["mathcode"]
# Extraction fixture: identical to mtp_acceptance_probe.py's control cell
# (verbatim-repeat passage, thinking off, 512 tokens) so rows are
# comparable with the banked extraction-512 numbers.
EXTRACT_PASSAGE = ("The lighthouse keeper checked the lamp at dusk. Salt wind "
                   "pressed against the tower glass. Ships passed far out. ") * 30
CELL_SETS["extraction"] = [
    ("extract-nothink-512",
     "Repeat the following passage exactly:\n\n" + EXTRACT_PASSAGE,
     False, 512),
]
CELLS = CELL_SETS[CELL_SET]


def get_metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {"drafts": 0.0, "accepted": 0.0, "steps": 0.0, "per_pos": {}}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            out["drafts"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            out["steps"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
            head, val = line.rsplit(" ", 1)
            pos = head.split('position="')[1].split('"')[0] \
                if 'position="' in head else "?"
            out["per_pos"][pos] = out["per_pos"].get(pos, 0.0) + float(val)
    return out


def chat(prompt, thinking, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read().decode())
    wall = time.perf_counter() - t0
    return d["usage"]["completion_tokens"], wall


rows = []
for name, prompt, thinking, mt in CELLS:
    name = name + TAG
    # warm run keeps the timed run free of first-touch effects per boot
    m0 = get_metrics()
    gen, wall = chat(prompt, thinking, mt)
    m1 = get_metrics()
    drafts = m1["drafts"] - m0["drafts"]
    acc = m1["accepted"] - m0["accepted"]
    steps = m1["steps"] - m0["steps"]
    pp = {p: m1["per_pos"].get(p, 0) - m0["per_pos"].get(p, 0)
          for p in m1["per_pos"]}
    ks = sorted(pp, key=lambda x: int(x) if x.isdigit() else 99)
    prof = "/".join(f"{pp[p]/steps*100:.0f}" for p in ks) if steps else ""
    ad = acc / drafts * 100 if drafts else 0.0
    tau = acc / steps if steps else 0.0
    ms_step = wall / steps * 1000 if steps else 0.0
    toks = gen / wall
    rows.append(f"{time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())},"
                f"{K},{name},{gen},{toks:.1f},{ad:.1f},{tau:.2f},"
                f"{ms_step:.1f},{prof}")
    print(f"k={K} {name}: {toks:.1f} tok/s acc={ad:.1f}% tau={tau:.2f} "
          f"ms/step={ms_step:.1f} pos%={prof}")

import os
new = not os.path.exists(OUT)
with open(OUT, "a") as f:
    if new:
        f.write("ts,k,cell,gen_tokens,tok_s,acc_pct,tau,ms_step,"
                "pos_acc_pct_profile\n")
    f.write("\n".join(rows) + "\n")
print(f"k={K} rows appended")
