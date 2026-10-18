"""EmbeddingGemma 2 vs the production MiniLM index on the Krishnamurti demo corpus.

Same 3,346 chunks, same 40 golden questions. Two things are measured, because
they are the two jobs the embedder does in this app:

  1. Retrieval: does top-5 land on the question's theme? There are no per-chunk
     relevance labels, so a theme keyword appearing in the retrieved text is the
     proxy. It is crude and identical for every model, so it compares fairly.
  2. Refusal: the app refuses a question when the top-1 cosine falls below a
     floor. That only works if in-corpus and off-corpus top-1 scores separate.

Also: 12 golden questions typed in Tamil, scored against the English corpus.
"""
import json, os, re, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA, CACHE = HERE / "data", HERE / "cache"
CACHE.mkdir(exist_ok=True)
DIMS = (768, 512, 256, 128)

THEME_RE = {
    "fear": r"\bfear", "authority": r"authorit|\bguru|teacher|follow",
    "self-knowledge": r"know(ing)? (your|one)sel|self-know|observer",
    "relationship": r"relationship", "awareness": r"aware|meditat|conditioning",
    "thought": r"\bthought|thinker|belief", "love": r"\blove", "comparison": r"compar",
    "truth": r"\btruth", "memory": r"\bmemor|the past", "freedom": r"\bfree",
}


def load():
    chunks = [json.loads(l) for l in open(DATA / "chunks.jsonl")]
    g = json.load(open(DATA / "golden_set.json"))
    audio_q = json.load(open(HERE / "audio" / "queries.json"))
    return chunks, g, audio_q


def theme_hit(chunks, idx, theme):
    pat = re.compile(THEME_RE[theme], re.I)
    return any(pat.search(chunks[i]["text"]) for i in idx)


def score(name, chunks, D, Q_in, themes, Q_off, off_types):
    """D: (n,d) normalised docs; Q_*: normalised queries."""
    s_in, s_off = Q_in @ D.T, Q_off @ D.T
    top_in, top_off = s_in.max(1), s_off.max(1)
    hits = [theme_hit(chunks, np.argsort(-s_in[i])[:5], themes[i]) for i in range(len(themes))]
    rand = np.array([t == "random" for t in off_types])
    # best single threshold separating in-corpus from random off-corpus
    cands = np.sort(np.concatenate([top_in, top_off[rand]]))
    best = max(((np.sum(top_in >= t) + np.sum(top_off[rand] < t)) / (len(top_in) + rand.sum()), t) for t in cands)
    auc = np.mean([[a > b for b in top_off[rand]] for a in top_in])
    people_above = int(np.sum(top_off[~rand] >= best[1]))
    return dict(model=name, theme_hit_at5=f"{sum(hits)}/{len(hits)}",
                in_top1_min=round(float(top_in.min()), 3), in_top1_mean=round(float(top_in.mean()), 3),
                rand_top1_max=round(float(top_off[rand].max()), 3), rand_top1_mean=round(float(top_off[rand].mean()), 3),
                gap=round(float(top_in.min() - top_off[rand].max()), 3), auc_in_vs_random=round(float(auc), 3),
                best_threshold=round(float(best[1]), 3), best_acc=round(float(best[0]), 3),
                named_person_qs_not_refused=f"{people_above}/{int((~rand).sum())}",
                misses=[i for i, h in enumerate(hits) if not h])


def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    import torch
    torch.set_num_threads(int(os.environ.get("THREADS", "3")))
    chunks, g, audio_q = load()
    q_in = [x["q"] for x in g["in_corpus"]]; themes = [x["theme"] for x in g["in_corpus"]]
    q_off = [x["q"] for x in g["off_corpus"]]; off_types = [x["type"] for x in g["off_corpus"]]
    ta = [x for x in audio_q if x["kind"] == "in"]
    results = []

    # --- baseline: production MiniLM vectors + its ONNX query embedder
    sys.path.insert(0, str(HERE))
    os.environ["MODEL_DIR"] = str(HERE / "models" / "all-MiniLM-L6-v2")
    from embedder import Embedder
    mini = Embedder()
    D0 = np.load(DATA / "vectors.npy")
    emb0 = lambda qs: norm(np.asarray(mini.encode(qs), dtype=np.float32))
    results.append(score("MiniLM-L6 384d (production)", chunks, D0, emb0(q_in), themes, emb0(q_off), off_types))
    Qta0 = emb0([x["ta"] for x in ta])
    results[-1]["tamil_typed_theme_hit_at5"] = f"{sum(theme_hit(chunks, np.argsort(-(Qta0 @ D0.T)[i])[:5], ta[i]['theme']) for i in range(len(ta)))}/{len(ta)}"

    # --- EmbeddingGemma 2, text-only load (270M)
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    m = SentenceTransformer("google/embeddinggemma-2", device="cpu",
                            model_kwargs={"torch_dtype": torch.float32},
                            config_kwargs={"vision_config": None, "audio_config": None})
    print(f"loaded text-only model in {time.time() - t0:.0f}s", flush=True)
    cache = CACHE / "eg2_docs_768.npy"
    if cache.exists():
        D = np.load(cache)
    else:
        docs = [f"title: {c['source']} | text: {c['text']}" for c in chunks]
        t0 = time.time()
        D = m.encode(docs, batch_size=16, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        el = time.time() - t0
        print(f"embedded {len(docs)} chunks in {el:.0f}s ({len(docs) / el:.1f} chunks/s)", flush=True)
        np.save(cache, D)
    enc_q = lambda qs: m.encode(qs, prompt_name="SearchQuery", convert_to_numpy=True, normalize_embeddings=True)
    Qi, Qo, Qta = enc_q(q_in), enc_q(q_off), enc_q([x["ta"] for x in ta])
    np.save(CACHE / "eg2_q_in.npy", Qi)
    for d in DIMS:
        r = score(f"EmbeddingGemma2 {d}d", chunks, norm(D[:, :d]), norm(Qi[:, :d]), themes, norm(Qo[:, :d]), off_types)
        Dd, Qt = norm(D[:, :d]), norm(Qta[:, :d])
        r["tamil_typed_theme_hit_at5"] = f"{sum(theme_hit(chunks, np.argsort(-(Qt @ Dd.T)[i])[:5], ta[i]['theme']) for i in range(len(ta)))}/{len(ta)}"
        results.append(r)

    json.dump(results, open(HERE / "results_text.json", "w"), indent=1)
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
