#!/usr/bin/env python3
"""Build the WP1 input bundle on jk-box: one {"id","doc"} JSON line per chunk, zstd + sha256.

Order and ids follow ~/krishnamurti-rag/index.py (c000000, c000001, …), so the GPU vectors line
up with the Chroma collection. The doc string matches the CPU eval: "title: <source> | text: <text>".
"""
import hashlib
import json
import os
import subprocess

SRC = os.path.expanduser("~/krishnamurti-rag/data/chunks.json")
OUT = os.path.expanduser("~/eg2-run/serve/chunks.jsonl.zst")

chunks = json.load(open(SRC))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
lines = "".join(json.dumps({"id": f"c{i:06d}", "doc": f"title: {c['source']} | text: {c['text']}"},
                           ensure_ascii=False) + "\n" for i, c in enumerate(chunks))
subprocess.run(["zstd", "-q", "-f", "-19", "-T3", "-o", OUT], input=lines.encode(), check=True)
digest = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
open(OUT + ".sha256", "w").write(f"{digest}  chunks.jsonl.zst\n")
print(f"{len(chunks)} chunks -> {os.path.getsize(OUT) / 1e6:.1f} MB, sha256 {digest[:16]}…")
