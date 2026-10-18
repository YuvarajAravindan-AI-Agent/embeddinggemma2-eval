#!/usr/bin/env python3
"""Temporary data door for one EmbeddingGemma 2 GPU run. Runs on jk-box, loopback only.

Caddy maps https://jk.ai-agentic-enterprises.com/eg2-<32-char secret>/* here with the prefix
stripped, so the secret path is the first lock and the X-Door-Token header the second.

  GET  /<file>            serve a file from SERVE_DIR (the input bundle, its sha256, the job script)
  PUT  /<run-id>/<file>   write into RUNS_DIR/<run-id>/<file> (results, job.log, progress.json)

Stdlib only. Started with systemd-run and stopped (with the Caddy route removed) after each run.
"""
import argparse
import hmac
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
MAX_PUT = 700 * 1024 * 1024


def make_handler(token, serve_dir, runs_dir):
    class H(BaseHTTPRequestHandler):
        def _deny(self, code, msg):
            self.send_response(code)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(msg.encode())

        def _authed(self):
            return hmac.compare_digest(self.headers.get("X-Door-Token", ""), token)

        def do_GET(self):
            name = self.path.lstrip("/")
            if not self._authed() or not NAME.match(name):
                return self._deny(403, "no")
            path = os.path.join(serve_dir, name)
            if not os.path.isfile(path):
                return self._deny(404, "missing")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(os.path.getsize(path)))
            self.end_headers()
            with open(path, "rb") as f:
                while chunk := f.read(1 << 20):
                    self.wfile.write(chunk)

        def do_PUT(self):
            parts = self.path.lstrip("/").split("/")
            n = int(self.headers.get("Content-Length") or -1)
            if not self._authed() or len(parts) != 2 or not all(NAME.match(p) for p in parts):
                return self._deny(403, "no")
            if n < 0 or n > MAX_PUT:
                return self._deny(413, "size")
            d = os.path.join(runs_dir, parts[0])
            os.makedirs(d, exist_ok=True)
            tmp = os.path.join(d, f".{parts[1]}.part")
            left = n
            with open(tmp, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            if left:
                os.remove(tmp)
                return self._deny(400, "short body")
            os.replace(tmp, os.path.join(d, parts[1]))
            self._deny(201, "ok")

        def log_message(self, fmt, *args):  # one line per request, no headers (the token rides in one)
            print(f"{self.command} {self.path.split('/')[-1]} -> {args[1] if len(args) > 1 else ''}", flush=True)

    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--serve-dir", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--port", type=int, default=8321)
    a = ap.parse_args()
    with open(a.token_file) as f:
        tok = f.read().strip()
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(tok, a.serve_dir, a.runs_dir)).serve_forever()
