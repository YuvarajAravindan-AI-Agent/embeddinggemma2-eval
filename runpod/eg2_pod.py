#!/usr/bin/env python3
"""Start/stop one short-lived RunPod GPU pod for an EmbeddingGemma 2 batch job, under a per-run cap.

Runs on the Contabo box (root). Stdlib only. Same pattern as /opt/medgemma-demo/medgemma_pod.py,
but a separate budget: MedGemma's ₹100/month cap does not cover these runs.

  eg2_pod.py up RUN_ID CAP_USD MINUTES   launch eg2-<RUN_ID>; refuses if worst case > CAP_USD
  eg2_pod.py down                        terminate every eg2-* pod
  eg2_pod.py status                      eg2-* pods + runtime phase + state
  eg2_pod.py spend                       month-to-date billed for eg2-* pods

Door settings come from /root/.config/eg2/door.json ({"url": …, "token_file": …}), written when
the jk-box data door is opened for a run.

Three independent stops: the job deletes its pod when done; a sleeper in the pod deletes it at
MINUTES; a systemd timer on this box runs `down` at MINUTES + 2.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v1"
KEY_FILE = "/root/.config/runpod/api_key"
HF_FILE = "/root/.config/huggingface/token"
DOOR_FILE = "/root/.config/eg2/door.json"
STATE = "/var/lib/eg2-run/pod.json"
UA = "eg2-pod/1.0 (+https://ai-agentic-enterprises.com)"  # RunPod Cloudflare 403s (1010) the default urllib UA

IMAGE = "pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime"
# versions from jk-box ~/eg2-eval/.venv, which already ran this model on CPU
PIP = "transformers==5.19.0 sentence-transformers==6.1.0 huggingface_hub==1.33.0 numpy==2.5.2 zstandard"
# 24 GB+ cards, best price per GB of VRAM first
GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA A40", "NVIDIA L4", "NVIDIA RTX A5000", "NVIDIA RTX A4500"]
MAX_USD_PER_HR = 0.60


def _key(path):
    with open(path) as f:
        return f.read().strip()


def api(method, path, body=None, query=""):
    req = urllib.request.Request(
        f"{API}{path}{query}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {_key(KEY_FILE)}", "Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raise SystemExit(f"RunPod API {method} {path}: HTTP {e.code} {e.read().decode()[:400]}")


def runtime(pid):
    q = ('query { pod(input: {podId: "%s"}) { desiredStatus runtime { uptimeInSeconds '
         'gpus { gpuUtilPercent memoryUtilPercent } } } }' % pid)
    try:
        req = urllib.request.Request("https://api.runpod.io/graphql", method="POST",
                                     data=json.dumps({"query": q}).encode(),
                                     headers={"Authorization": f"Bearer {_key(KEY_FILE)}",
                                              "Content-Type": "application/json", "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            pod = (json.loads(r.read()).get("data") or {}).get("pod") or {}
    except Exception as e:
        return f"runtime=? ({type(e).__name__})"
    rt = pod.get("runtime")
    if not rt:
        return f"{pod.get('desiredStatus')} no runtime yet (pulling image / waiting for host)"
    g = (rt.get("gpus") or [{}])[0]
    return f"{pod.get('desiredStatus')} up={rt.get('uptimeInSeconds')}s gpu={g.get('gpuUtilPercent')}% vram={g.get('memoryUtilPercent')}%"


def _write_state(d):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, STATE)


def _read_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def eg2_pods():
    return [p for p in (api("GET", "/pods") or []) if p.get("name", "").startswith("eg2-")]


def cmd_spend():
    now = dt.datetime.now(dt.timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    q = f"?bucketSize=month&grouping=podId&startTime={start:%Y-%m-%dT%H:%M:%SZ}&endTime={now:%Y-%m-%dT%H:%M:%SZ}"
    rows = api("GET", "/billing/pods", query=q) or []
    total = sum(float(r.get("amount") or 0) for r in rows)
    print(f"account month-to-date billed: ${total:.3f} (all pods; console lags ~1 h)")
    return total


def cmd_up(run_id, cap_usd, minutes):
    cap_usd, minutes = float(cap_usd), int(minutes)
    if not 10 <= minutes <= 180:
        raise SystemExit("MINUTES must be 10..180")
    if eg2_pods():
        raise SystemExit("an eg2 pod is already on the account; run `down` first")
    worst = MAX_USD_PER_HR * (minutes + 2) / 60
    if worst > cap_usd:
        raise SystemExit(f"REFUSED: worst case ${worst:.2f} (${MAX_USD_PER_HR}/hr × {minutes + 2} min) > cap ${cap_usd:.2f}")
    door = json.load(open(DOOR_FILE))
    deadline = time.time() + minutes * 60
    killer = ("import os,time,urllib.request as u\n"
              f"time.sleep({minutes * 60})\n"
              "pid=os.environ['RUNPOD_POD_ID'];h={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'],'User-Agent':'eg2-killer/1.0'}\n"
              "for m,p in (('DELETE',''),('POST','/stop')):\n"
              "  try: u.urlopen(u.Request('https://rest.runpod.io/v1/pods/'+pid+p,method=m,headers=h),timeout=30); break\n"
              "  except Exception as e: print('self-terminate',m,e,flush=True)\n")
    fetch = ("import os,urllib.request as u\n"
             "r=u.Request(os.environ['DOOR']+'/eg2_job.py',headers={'X-Door-Token':os.environ['DOOR_TOKEN'],'User-Agent':'eg2-boot/1.0'})\n"
             "open('/eg2_job.py','wb').write(u.urlopen(r,timeout=120).read())\n")
    start = (f"python3 -c \"$KILLER\" & "
             f"pip install -q --no-cache-dir {PIP} && python3 -c \"$FETCH\" && exec python3 /eg2_job.py")
    body = {
        "name": f"eg2-{run_id}",
        "imageName": IMAGE,
        "cloudType": "COMMUNITY",
        "gpuTypeIds": GPUS, "gpuTypePriority": "custom", "gpuCount": 1,
        "allowedCudaVersions": ["12.6", "12.7", "12.8", "12.9", "13.0"],
        "containerDiskInGb": 40, "volumeInGb": 0,
        "env": {"HF_TOKEN": _key(HF_FILE), "DOOR": door["url"], "DOOR_TOKEN": _key(door["token_file"]),
                "RUN_ID": run_id, "DEADLINE": str(int(deadline)), "KILLER": killer, "FETCH": fetch},
        "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [start],
    }
    pod = api("POST", "/pods", body)
    pid, rate = pod["id"], float(pod.get("costPerHr") or pod.get("adjustedCostPerHr") or 0)
    gpu = (pod.get("machine") or {}).get("gpuTypeId") or (pod.get("gpu") or {}).get("id") or "?"
    if rate > MAX_USD_PER_HR:
        api("DELETE", f"/pods/{pid}")
        raise SystemExit(f"REFUSED: got {gpu} at ${rate}/hr > ${MAX_USD_PER_HR}; pod terminated")
    _write_state({"pod_id": pid, "run_id": run_id, "gpu": gpu, "cost_per_hr": rate, "cap_usd": cap_usd,
                  "started": time.time(), "deadline": deadline, "status": "running"})
    subprocess.run(["systemd-run", "--unit", f"eg2-down-{pid}", f"--on-active={minutes * 60 + 120}",
                    sys.executable, os.path.abspath(__file__), "down"], check=False, capture_output=True)
    print(f"pod {pid} eg2-{run_id} on {gpu} at ${rate}/hr; worst case ${rate * (minutes + 2) / 60:.2f} "
          f"(cap ${cap_usd:.2f}); hard stop in {minutes} min")


def cmd_down():
    gone = []
    for p in eg2_pods():
        api("DELETE", f"/pods/{p['id']}")
        gone.append(p["id"])
    if (s := _read_state()) and s.get("status") == "running":
        s.update(status="off", stopped=time.time(),
                 est_cost_usd=round(s["cost_per_hr"] * (time.time() - s["started"]) / 3600, 3))
        _write_state(s)
    print("terminated:", ", ".join(gone) or "nothing running")


def cmd_status():
    for p in eg2_pods():
        print(f"{p['id']}  {p.get('name')!r:24} {p.get('desiredStatus'):9} ${p.get('costPerHr')}/hr  {runtime(p['id'])}")
    s = _read_state() or {}
    if s.get("status") == "running":
        s["elapsed_min"] = round((time.time() - s["started"]) / 60, 1)
        s["est_cost_usd"] = round(s["cost_per_hr"] * (time.time() - s["started"]) / 3600, 3)
    print("state:", s)


if __name__ == "__main__":
    a = sys.argv[1:] or ["status"]
    {"up": lambda: cmd_up(*a[1:4]), "down": cmd_down, "status": cmd_status, "spend": cmd_spend}[a[0]]()
