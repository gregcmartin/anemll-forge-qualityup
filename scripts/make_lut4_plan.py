#!/usr/bin/env python3
"""Write a quantization plan: release 0.2's plan with chosen MLP layers' matrices at LUT4.

    python scripts/make_lut4_plan.py --layers 24-63 --out configs/quantization/mix25in_vq3pA_lut4L24-63.json
    python scripts/make_lut4_plan.py --layers 32-63 --mats gate,up --out /tmp/plan.json   # gate/up only

Release 0.2 (configs/quantization/mix25in_vq3pA.json) quantizes every MLP matrix as `vector 2x64 + pcs` (three bits
per weight). LUT4 (`LUT4 per-tensor + pcs`, four bits) cuts a matrix's GPTQ error to about a third of 2x64's
(docs/QUANTIZATION.md, "Why three bits in every MLP matrix"). The late layers carry most of the MLP error
(QUANTIZATION_NOTES.md, in-domain MLP band sweep: layers 24-63 hold 89%).

Cost on M6 per LUT4 MLP layer, against 2x64: +0.22 ms per verify call. Prefill speed is unchanged. Extra memory per
layer is about 33 MB of weights, plus 44.6 MB when down_proj is LUT4, because a LUT4 down_proj is compiled twice in
packages that hold decode and prefill functions.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/quantization/mix25in_vq3pA.json"
LUT4 = "LUT4 per-tensor + pcs"


def layers(spec):
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    if not out or min(out) < 0 or max(out) > 63:
        raise SystemExit(f"layers must be within 0-63: {spec}")
    return sorted(out)


def make(base, chosen, mats):
    plan = dict(base)
    for i in chosen:
        for m in mats:
            key = f"{i}.{m}"
            if key not in plan:
                raise SystemExit(f"{key} is not in the base plan")
            plan[key] = LUT4
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--layers", required=True, help="MLP layers at LUT4, e.g. 24-63 or 32-47,56-63")
    p.add_argument("--mats", default="gate,up,down", help="MLP matrices to upgrade (default: gate,up,down)")
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    mats = [m for m in a.mats.split(",") if m]
    if not set(mats) <= {"gate", "up", "down"}:
        raise SystemExit("--mats takes gate, up and/or down")
    chosen = layers(a.layers)
    plan = make(json.loads(a.base.read_text()), chosen, mats)
    a.out.write_text(json.dumps(plan, indent=1) + "\n")
    mb = len(chosen) * (len(mats) / 3 * 33.4 + (44.6 if "down" in mats else 0))
    print(f"{a.out}: {len(chosen)} layers x {','.join(mats)} at LUT4; estimated +{mb / 1000:.1f} GB ANE memory, "
          f"+{0.22 * len(chosen) * len(mats) / 3:.1f} ms per verify (M6)")


if __name__ == "__main__":
    main()
