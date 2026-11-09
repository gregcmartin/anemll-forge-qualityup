"""Four-way speed harness (Splash, original ANEMLL, latest fork, qualityup): identical frozen prompts, greedy,
256 output tokens, thinking off, one request at a time. Rates are each server's own: ANEMLL's /health prefill and
decode stats after each request, Splash's response timings. Responses are kept for later comparison.
    python m6_bench4.py <deployment> <anemll|splash> <prompts.json> <out csv> <responses jsonl> [cases]"""
import csv, json, sys, time, urllib.request
from pathlib import Path
name, kind, prompts, out, resp_out = sys.argv[1], sys.argv[2], json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4]), Path(sys.argv[5])
cases = sys.argv[6].split(",") if len(sys.argv) > 6 else None
if kind == "splash":
    import os
    URL = os.environ.get("SPLASH_URL", "http://127.0.0.1:8000")
    MODEL = os.environ.get("SPLASH_MODEL", "jaromer/Qwen3.8-27B-Uncensored-GGUF:Q6_K")
    HEAD = {"Authorization": "Bearer " + Path(os.environ["SPLASH_API_KEY_FILE"]).read_text().strip()}
else:
    URL, MODEL, HEAD = "http://127.0.0.1:8765", "qwen38-27b-ane", {}


def call(path, body=None, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json", **HEAD})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


new = not out.exists()
with out.open("a", newline="") as fh, resp_out.open("a") as rf:
    w = csv.writer(fh)
    if new:
        w.writerow(["deployment", "case", "trial", "prompt_sha256", "prompt_tokens", "output_tokens", "prefill_tps",
                    "decode_tps", "request_s", "cached_tokens", "tok_per_verify", "accept_pct"])
    for r in prompts:
        if cases and r["case"] not in cases:
            continue
        body = {"model": MODEL, "messages": [{"role": "user", "content": r["prompt"]}], "max_tokens": 256,
                "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.perf_counter()
        resp = call("/v1/chat/completions", body)
        dt = time.perf_counter() - t0
        u = resp.get("usage", {})
        if kind == "splash":
            t, sp = resp["timings"], resp.get("metrics", {}).get("speculation", {})
            pre, dec, cached = t["prompt_per_second"], t["predicted_per_second"], t.get("cache_n", 0)
            cyc = sp.get("cycles") or 0
            tpv = t["predicted_n"] / cyc if cyc else ""
            acc = 100 * sp["accepted_draft_tokens"] / sp["drafted_tokens"] if sp.get("drafted_tokens") else ""
        else:
            h = call("/health")
            p, d = h["prefill"], h["decode"]
            pre, dec, cached = p["tokens_per_second"], d["tokens_per_second"], p.get("cached_tokens", 0)
            tpv, acc = d.get("tokens_per_call") or "", d.get("draft_accept_percent") or ""
        row = [name, r["case"], r["trial"], r["sha256"], u.get("prompt_tokens"), u.get("completion_tokens"), pre, dec,
               dt, cached, tpv, acc]
        w.writerow(row)
        fh.flush()
        rf.write(json.dumps({"deployment": name, "case": r["case"], "trial": r["trial"], "prompt_sha256": r["sha256"],
                             "content": resp["choices"][0]["message"].get("content"),
                             "finish_reason": resp["choices"][0].get("finish_reason")}) + "\n")
        rf.flush()
        print(json.dumps(dict(zip(["deployment", "case", "trial", "sha", "prompt_tokens", "output_tokens", "prefill_tps",
                                   "decode_tps", "request_s", "cached", "tok_per_verify", "accept_pct"], row))), flush=True)
