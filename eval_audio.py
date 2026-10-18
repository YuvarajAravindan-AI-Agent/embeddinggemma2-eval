"""Spoken questions (English + Tamil) searched directly against the TEXT index — no transcription.

Needs cache/eg2_docs_768.npy from eval_text.py. Each spoken question is compared
with the same question typed, so the audio cost is visible on its own.
"""
import json, os, re
from pathlib import Path

import numpy as np
import librosa

from eval_text import HERE, CACHE, DATA, DIMS, theme_hit, norm


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(int(os.environ.get("THREADS", "3")))
    chunks = [json.loads(l) for l in open(DATA / "chunks.jsonl")]
    qs = json.load(open(HERE / "audio" / "queries.json"))
    D = np.load(CACHE / "eg2_docs_768.npy")
    dtype = torch.bfloat16 if os.environ.get("DTYPE") == "bf16" else torch.float32
    m = SentenceTransformer("google/embeddinggemma-2", device="cpu",
                            model_kwargs={"torch_dtype": dtype}, config_kwargs={"vision_config": None})

    def wav(path):
        y, _ = librosa.load(path, sr=16000, mono=True)
        return {"array": y, "sampling_rate": 16000}

    rows = {}  # variant -> list of (id, kind, theme, 768d query vector)
    for x in qs:
        variants = {"text_en": x["en"], "audio_en": wav(HERE / "audio" / f"{x['id']}_en.wav")}
        if x["ta"]:
            variants |= {"text_ta": x["ta"], "audio_ta": wav(HERE / "audio" / f"{x['id']}_ta.wav")}
        for v, inp in variants.items():
            kw = {"prompt_name": "SearchQuery"} if v.startswith("text") else {}
            e = m.encode([inp], convert_to_numpy=True, normalize_embeddings=True, **kw)[0]
            rows.setdefault(v, []).append((x["id"], x["kind"], x["theme"], e))
        print("encoded", x["id"], flush=True)

    out = []
    off_audio = np.array([e for _, k, _, e in rows["audio_en"] if k == "off"])
    for d in DIMS:
        Dd = norm(D[:, :d])
        off_top = (norm(off_audio[:, :d]) @ Dd.T).max(1)
        for v, lst in rows.items():
            ins = [(i, t, e) for i, k, t, e in lst if k == "in"]
            Q = norm(np.array([e for *_, e in ins])[:, :d])
            S = Q @ Dd.T
            hits = sum(theme_hit(chunks, np.argsort(-S[j])[:5], ins[j][1]) for j in range(len(ins)))
            top = S.max(1)
            r = dict(dim=d, query=v, theme_hit_at5=f"{hits}/{len(ins)}",
                     in_top1_mean=round(float(top.mean()), 3), in_top1_min=round(float(top.min()), 3))
            if v.startswith("audio"):
                r["spoken_offtopic_top1_max"] = round(float(off_top.max()), 3)
                r["separable_from_spoken_offtopic"] = bool(top.min() > off_top.max())
            out.append(r)
            print(json.dumps(r), flush=True)
        # agreement: does spoken query retrieve the same top-5 as typed?
        for lang in ("en", "ta"):
            T = norm(np.array([e for _, k, _, e in rows[f"text_{lang}"] if k == "in"])[:, :d]) @ Dd.T
            A = norm(np.array([e for _, k, _, e in rows[f"audio_{lang}"] if k == "in"])[:, :d]) @ Dd.T
            ov = np.mean([len(set(np.argsort(-T[j])[:5]) & set(np.argsort(-A[j])[:5])) / 5 for j in range(len(T))])
            out.append(dict(dim=d, query=f"audio_{lang} vs text_{lang}", top5_overlap=round(float(ov), 2)))
            print(json.dumps(out[-1]), flush=True)
    # show what spoken Tamil retrieves for two questions, at 768d
    for j, (i, k, t, e) in enumerate(rows["audio_ta"][:12]):
        if i in ("fear1", "rel1"):
            top = np.argsort(-(e @ D.T))[:3]
            print(f"\n[{i} spoken Tamil] ->", *[f"\n  {chunks[c]['source']}: {chunks[c]['text'][:140]!r}" for c in top])
    json.dump(out, open(HERE / "results_audio.json", "w"), indent=1)


if __name__ == "__main__":
    main()
