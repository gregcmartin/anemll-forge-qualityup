"""Freeze the four-way benchmark prompts (same construction as m6_bench.py): 2K and 31.5K-token slices of this
repository's Markdown (prose) and Python (coding), 3 trials each, each opened by a unique request line.
    python m6_prompts.py <src dir> <tokenizer.json> <out prompts.json>"""
import hashlib, json, sys
from pathlib import Path
from tokenizers import Tokenizer
src, tok, out = Path(sys.argv[1]), Tokenizer.from_file(sys.argv[2]), Path(sys.argv[3])
files = lambda pats: sorted({p for pat in pats for p in src.glob(pat) if p.is_file()})
prose = "\n\n".join(p.read_text() for p in files(["*.md", "docs/*.md", "docs/**/*.md"]))
code = "\n\n".join(f"# file: {p.relative_to(src)}\n{p.read_text()}" for p in files(["scripts/*.py", "coreai/*.py"]))
ids = {"prose": tok.encode(prose).ids, "coding": tok.encode(code).ids}
SIZES = {"short": 2048, "long": 31500}
ASK = {"prose": "Summarize the main points of the following documentation in a few paragraphs.",
       "coding": "Explain what the following Python code does, then point out any bugs you see."}
rows = []
for case in ("short-prose", "short-coding", "long-prose", "long-coding"):
    size, kind = case.split("-")
    n = SIZES[size]
    for trial in (1, 2, 3):
        start = (trial - 1) * n if size == "short" else 3 * SIZES["short"] + (trial - 1) * n
        body = tok.decode(ids[kind][start:start + n - 60])
        head = f"Request {hashlib.sha256(f'{case}/{trial}'.encode()).hexdigest()[:16]}. {ASK[kind]}\n\n"
        p = head + (f"```python\n{body}\n```" if kind == "coding" else body)
        rows.append(dict(case=case, trial=trial, prompt=p, sha256=hashlib.sha256(p.encode()).hexdigest()))
out.write_text(json.dumps(rows))
print(len(rows), "prompts ->", out)
