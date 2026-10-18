"""
all-MiniLM-L6-v2 embeddings via ONNX Runtime.

This replaces the Ollama `all-minilm` call the Netcup original made. It is
deliberately the *same underlying model*, not a similar one: the retrieval
guardrail MIN_RELEVANCE = 0.5 was calibrated empirically against Ollama's
vectors, and a different embedder would silently invalidate it. See
ingest/verify_embeddings.py for the measurement that backs that claim.

ONNX Runtime rather than torch/sentence-transformers because the AppSail
image has to cold-start inside a 30s request budget — torch alone is close
to a gigabyte, and none of it is needed for a 6-layer encoder.

Pooling is mean-over-non-padding-tokens followed by L2 normalisation, which
is what the sentence-transformers config for this model specifies and what
Ollama does. Because vectors come back normalised, cosine similarity is a
plain dot product.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "models" / "all-MiniLM-L6-v2"))

# all-minilm truncates at 256 tokens and says nothing when it does. chunk.py
# sizes chunks to stay inside that window; enforcing it here too means a
# hand-fed oversized chunk fails the same way in both places rather than
# being quietly cut in only one of them.
MAX_TOKENS = 256


class Embedder:
    def __init__(self, model_dir: Path | str = MODEL_DIR):
        model_dir = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        opts = ort.SessionOptions()
        # One thread per request keeps latency predictable under AppSail's
        # concurrency rather than letting a single query eat every core.
        opts.intra_op_num_threads = int(os.environ.get("ONNX_THREADS", "2"))
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch_size):
            out.append(self._encode_batch(texts[start:start + batch_size]))
        return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)

        hidden = self.session.run(None, feed)[0]  # (batch, seq, 384)

        m = mask[..., None].astype(np.float32)
        summed = (hidden * m).sum(axis=1)
        counts = np.clip(m.sum(axis=1), 1e-9, None)
        pooled = summed / counts

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
