"""Summarise the four-way run: medians of three per workload (as 8 November), verify-call time, memory, and
byte-identical responses between deployments.   python m6_analyze4.py <bench4 dir> <out dir>"""
import csv, json, statistics as st, sys
from pathlib import Path
d, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(d / "bench4.csv")))
DEPS = ["splash", "original", "fork", "qualityup"]
CASES = ["short-prose", "short-coding", "long-prose", "long-coding"]
LABEL = {"short-prose": "2K prose", "short-coding": "2K coding", "long-prose": "31.5K prose", "long-coding": "31.5K coding"}
med = lambda xs: st.median(xs) if xs else float("nan")
summary = []
for c in CASES:
    for dep in DEPS:
        rs = [r for r in rows if r["deployment"] == dep and r["case"] == c]
        assert len(rs) == 3, (dep, c, len(rs))
        assert all(int(float(r["cached_tokens"] or 0)) == 0 for r in rs), (dep, c)
        f = lambda k: [float(r[k]) for r in rs if r[k] not in ("", None)]
        vms = [1000 * float(r["tok_per_verify"]) / float(r["decode_tps"]) for r in rs if r["tok_per_verify"]]
        summary.append(dict(workload=LABEL[c], deployment=dep, prefill_tps=med(f("prefill_tps")), decode_tps=med(f("decode_tps")),
                            request_s=med(f("request_s")), tok_per_verify=med(f("tok_per_verify")), verify_ms=med(vms),
                            prompt_tokens=med(f("prompt_tokens"))))
with open(out / "summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
print(f"{'workload':13} {'metric':16}" + "".join(f"{x:>11}" for x in DEPS))
for c in CASES:
    for key, name, fmt in (("prefill_tps", "prefill tok/s", "{:11.1f}"), ("decode_tps", "decode tok/s", "{:11.2f}"),
                           ("request_s", "request s", "{:11.2f}"), ("tok_per_verify", "tok/verify", "{:11.2f}"),
                           ("verify_ms", "ms per verify", "{:11.1f}")):
        vals = [next(s[key] for s in summary if s["workload"] == LABEL[c] and s["deployment"] == dep) for dep in DEPS]
        print(f"{LABEL[c]:13} {name:16}" + "".join(fmt.format(v) for v in vals))
# memory (ANEMLL runs): wired above the lowest no-model baseline + peak worker footprint
mem = [r for r in csv.reader(open(d / "mem.csv")) if len(r) >= 4]
base = min(float(r[3]) for r in mem if r[2] == "baseline")
memsum = []
for dep in DEPS[1:]:
    rs = [r for r in mem if r[1] == dep]
    loaded = next(float(r[3]) for r in rs if r[2] == "loaded")
    pw = max(float(r[3]) for r in rs if r[2] in ("loaded", "sample"))
    pf = max(float(r[4]) for r in rs if r[2] in ("loaded", "sample") and r[4]) / 1024
    sw = max(float(r[5]) for r in rs if r[2] == "sample" and len(r) > 5 and r[5]) / 1024
    memsum.append(dict(deployment=dep, baseline_gb=base, loaded_wired_gb=loaded, peak_wired_gb=pw, peak_footprint_gb=round(pf, 2),
                       total_gb=round(pw - base + pf, 2), peak_swap_gb=round(sw, 2)))
with open(out / "memory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(memsum[0])); w.writeheader(); w.writerows(memsum)
for m in memsum: print(m)
splash_res = [r for r in mem if r[1] == "splash"]
print("splash resident wired", splash_res[0][3] if splash_res else None)
# byte-identical responses
resp = [json.loads(l) for l in open(d / "responses.jsonl") if l.strip()]
by = {(r["deployment"], r["prompt_sha256"]): r["content"] for r in resp}
shas = sorted({r["prompt_sha256"] for r in resp})
for a, b in (("original", "fork"), ("fork", "qualityup"), ("original", "qualityup"), ("splash", "fork")):
    same = sum(by.get((a, s)) is not None and by.get((a, s)) == by.get((b, s)) for s in shas)
    print(f"identical responses {a} vs {b}: {same}/{len(shas)}")
