#!/usr/bin/env python3
"""WP1: embed the full Krishnamurti archive with EmbeddingGemma 2 on a RunPod GPU.

Runs inside the pod. Pulls its input from the jk-box data door, pushes results back,
then deletes its own pod. Every HTTP call carries a custom User-Agent (RunPod's
Cloudflare 403s the urllib default, and the door would be no different behind Caddy).

Env: DOOR (https://…/eg2-<secret>), DOOR_TOKEN, RUN_ID, HF_TOKEN, DEADLINE (unix time the
controller will kill us), RUNPOD_POD_ID + RUNPOD_API_KEY (injected by RunPod).

Doc strings match the CPU eval exactly ("title: <source> | text: <text>", text-only model
load, normalised), so GPU vectors can be checked against the CPU ones.
"""
import hashlib
import io
import json
import os
import threading
import time
import traceback
import urllib.request

UA = "eg2-job/1.0 (+https://ai-agentic-enterprises.com)"
DOOR, TOKEN, RUN = os.environ["DOOR"].rstrip("/"), os.environ["DOOR_TOKEN"], os.environ["RUN_ID"]
DEADLINE = float(os.environ["DEADLINE"])
BATCH = int(os.environ.get("BATCH", "128"))
LOG = io.StringIO()
progress = {"phase": "start", "done": 0, "total": None, "rate": None, "t0": time.time()}


def log(msg):
    line = f"[{time.time() - progress['t0']:6.0f}s] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")


def http(method, url, data=None, headers=None, timeout=600):
    h = {"User-Agent": UA, **(headers or {})}
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, method=method, headers=h),
                                        timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if attempt == 3:
                raise
            print(f"{method} {url.rsplit('/', 1)[-1]} failed ({e}); retrying", flush=True)
            time.sleep(5 * (attempt + 1))


def door_get(name):
    return http("GET", f"{DOOR}/{name}", headers={"X-Door-Token": TOKEN})


def door_put(name, data):
    http("PUT", f"{DOOR}/{RUN}/{name}", data=data,
         headers={"X-Door-Token": TOKEN, "Content-Type": "application/octet-stream"})


def push_status():
    try:
        door_put("progress.json", json.dumps({**progress, "at": time.time()}).encode())
        door_put("job.log", LOG.getvalue().encode())
    except Exception as e:
        print("status push failed:", e, flush=True)


def heartbeat():
    while True:
        time.sleep(60)
        push_status()


def self_delete():
    pid, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if not pid or not key:
        return log("no RUNPOD_POD_ID/RUNPOD_API_KEY; controller timer will tear down")
    for method, suffix in (("DELETE", ""), ("POST", "/stop")):
        try:
            http(method, f"https://rest.runpod.io/v1/pods/{pid}{suffix}",
                 headers={"Authorization": f"Bearer {key}"}, timeout=30)
            return
        except Exception as e:
            print("self-delete", method, e, flush=True)


def main():
    import numpy as np
    import torch
    import zstandard
    from sentence_transformers import SentenceTransformer

    progress["phase"] = "download"
    want = door_get("chunks.jsonl.zst.sha256").decode().split()[0]
    raw = door_get("chunks.jsonl.zst")
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise SystemExit(f"bundle sha256 mismatch: {got} != {want}")
    rows = [json.loads(l) for l in zstandard.decompress(raw).decode().splitlines() if l]
    ids, docs = [r["id"] for r in rows], [r["doc"] for r in rows]
    progress["total"] = len(docs)
    log(f"bundle ok: {len(docs)} chunks, {len(raw) / 1e6:.1f} MB; GPU {torch.cuda.get_device_name(0)}")

    progress["phase"] = "load"
    t = time.time()
    m = SentenceTransformer("google/embeddinggemma-2", device="cuda", token=os.environ["HF_TOKEN"],
                            model_kwargs={"torch_dtype": torch.bfloat16},
                            config_kwargs={"vision_config": None, "audio_config": None})
    log(f"model loaded in {time.time() - t:.0f}s")

    # length-sorted batches waste less padding; the original order is restored afterwards
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
    out = np.zeros((len(docs), 768), dtype=np.float16)
    progress["phase"] = "embed"
    t0 = time.time()
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        v = m.encode([docs[i] for i in idx], batch_size=BATCH, convert_to_numpy=True, normalize_embeddings=True)
        out[idx] = v.astype(np.float16)
        done = s + len(idx)
        el = time.time() - t0
        progress.update(done=done, rate=round(done / el, 1) if el else None)
        if done == len(idx) or done % (BATCH * 40) < BATCH or done == len(order):
            eta = (len(order) - done) / (done / el) if el else 0
            log(f"{done}/{len(order)} chunks, {done / el:.1f}/s, eta {eta / 60:.1f} min")
            # stop early rather than be killed mid-run: the projection says we cannot finish in time
            if el > 120 and time.time() + eta * 1.15 > DEADLINE - 180:
                raise SystemExit(f"projected finish past the deadline (eta {eta / 60:.0f} min); stopping")
    log(f"embedded {len(order)} chunks in {time.time() - t0:.0f}s")

    progress["phase"] = "upload"
    norms = np.linalg.norm(out.astype(np.float32), axis=1)
    buf = io.BytesIO()
    np.save(buf, out)
    data = buf.getvalue()
    meta = {"rows": len(ids), "dim": 768, "dtype": "float16", "model": "google/embeddinggemma-2",
            "doc_format": "title: <source> | text: <text>", "norm_min": float(norms.min()),
            "norm_max": float(norms.max()), "sha256": hashlib.sha256(data).hexdigest(),
            "gpu": torch.cuda.get_device_name(0), "embed_seconds": round(time.time() - t0),
            "torch": torch.__version__}
    door_put("ids.json", json.dumps(ids).encode())
    door_put("eg2_768.f16.npy", data)
    door_put("meta.json", json.dumps(meta, indent=2).encode())
    log(f"uploaded {len(data) / 1e6:.0f} MB; norms {meta['norm_min']:.4f}..{meta['norm_max']:.4f}")
    progress["phase"] = "done"


if __name__ == "__main__":
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        main()
    except BaseException as e:
        progress["phase"] = "failed"
        log("FAILED: " + "".join(traceback.format_exception(e))[-1500:])
    finally:
        push_status()
        self_delete()
        time.sleep(600)  # if the delete failed, idle quietly until the controller's timer removes us
