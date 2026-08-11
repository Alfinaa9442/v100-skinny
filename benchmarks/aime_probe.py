"""Matched long-reasoning fixture (AIME-class): competition math with
thinking mode, 15k+ token traces — the external stack's long_decode
workload shape. Greedy (our losslessness regime; theirs samples at
temperature — noted caveat). Reports decode-only tok/s (streaming),
acceptance, tokens/round, and pos1-3 rates for the k=3-equivalent
comparison. Usage: aime_probe.py <k>"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.6-27b-nvfp4-skinny"
K = int(sys.argv[1])
TEMP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
MAXTOK = 18000

PROBLEMS = [
    ("aime_a",
     "Solve this competition problem with complete rigor, verifying every "
     "step and exploring the full solution space before concluding.\n\n"
     "Find the number of ordered pairs (a, b) of positive integers with "
     "a, b <= 2026 such that a^2 + b^2 is divisible by ab + 1."),
    ("aime_b",
     "Solve this competition problem with complete rigor, verifying every "
     "step and exploring the full solution space before concluding.\n\n"
     "Let S be the set of positive integers n <= 10000 for which "
     "sigma(n) (the sum of divisors of n) is a power of 2. Determine "
     "the sum of all elements of S."),
    ("aime_c",
     "Solve this competition problem with complete rigor, verifying every "
     "step and exploring the full solution space before concluding.\n\n"
     "A frog starts at 0 on the number line. Each second it jumps +1 "
     "with probability 2/3 or -1 with probability 1/3, independently. "
     "Let p be the probability it ever reaches +5 without first touching "
     "-3. Compute p exactly as a reduced fraction m/n and give m + n."),
]


def met():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {"d": 0.0, "a": 0.0, "s": 0.0, "pp": {}}
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            out["d"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["a"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            out["s"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
            head, val = line.rsplit(" ", 1)
            pos = head.split('position="')[1].split('"')[0]
            out["pp"][pos] = out["pp"].get(pos, 0.0) + float(val)
    return out


def stream(prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMP, "top_p": 0.95 if TEMP > 0 else 1.0,
        "max_completion_tokens": MAXTOK,
        "chat_template_kwargs": {"enable_thinking": True},
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"})
    tf = tl = None
    gen = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            t = time.perf_counter()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                gen = d["usage"]["completion_tokens"]
                continue
            ch = d.get("choices")
            if ch and (ch[0]["delta"].get("content")
                       or ch[0]["delta"].get("reasoning")):
                if tf is None:
                    tf = t
                tl = t
    return gen, tf, tl


print(f"{'fixture':>8} {'gen_tok':>8} {'decode tok/s':>12} {'acc%':>6} "
      f"{'tok/round':>9} {'pos1':>5} {'pos2':>5} {'pos3':>5} "
      f"{'3pos-mean':>9}")
for name, prompt in PROBLEMS:
    m0 = met()
    gen, tf, tl = stream(prompt)
    m1 = met()
    steps = m1["s"] - m0["s"]
    acc = m1["a"] - m0["a"]
    drafts = m1["d"] - m0["d"]
    tau = acc / steps if steps else 0.0
    span = tl - tf if tf and tl else 0.0
    dtoks = (gen - (tau + 1)) / span if span > 0 else 0.0
    p = {i: (m1["pp"].get(str(i), 0) - m0["pp"].get(str(i), 0)) / steps * 100
         for i in (1, 2, 3)}
    print(f"{name:>8} {gen:>8} {dtoks:>12.1f} "
          f"{acc/max(drafts,1)*100:>6.1f} {tau+1:>9.2f} "
          f"{p[1]:>5.0f} {p[2]:>5.0f} {p[3]:>5.0f} "
          f"{(p[1]+p[2]+p[3])/3:>9.1f}", flush=True)
print(f"AIME_PROBE_DONE k={K} temp={TEMP}")
