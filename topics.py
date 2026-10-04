"""Topic models of a set of texts (or of the passages of one text) with a choice of method.

All methods start from the same bag of words: the words of each document without stop words, rare
words and words that are in most documents. They differ in what a topic is:

  lda     Latent Dirichlet Allocation: a topic is a probability distribution over words and a
          document a mix of topics.
  nmf     Non-negative Matrix Factorization of the TF-IDF matrix: documents as sums of topics.
  lsi     Latent Semantic Indexing (truncated SVD of the TF-IDF matrix): topics are directions in
          word space and can weigh words negatively.
  hdp     Hierarchical Dirichlet Process: like LDA, but it chooses the number of topics itself.
  kmeans  k-means clustering of the TF-IDF vectors: every document belongs to exactly one
          cluster, named by the words that weigh most in its centre.

The output has the same format as MD-bertopic's topics.json, plus a coherence score (c_v) that
compares runs: higher means the top words of each topic tend to occur together in the texts.
"""

import json
import logging
from uuid import uuid4

import numpy as np
from fastapi import HTTPException

from text import bool_param, int_param, languages_from, now, stem, stop_words, words, words_with_spans

logger = logging.getLogger("md-gensim")

TOPICS_FORMAT = "messydesk-topics/1"
METHODS = {
    "lda": "LDA (Latent Dirichlet Allocation)",
    "nmf": "NMF (Non-negative Matrix Factorization)",
    "lsi": "LSI (Latent Semantic Indexing)",
    "hdp": "HDP (Hierarchical Dirichlet Process)",
    "kmeans": "k-means clustering",
}
MIN_DOCUMENTS = 2
TOP_WORDS = 10
TAG_MIN_SHARE = 0.3  # a document is tagged with topics holding at least this share of it
TAG_MAX_PER_PAGE = 3
COHERENCE_MAX_DOCS = 5000
HDP_MIN_SHARE = 0.1


# ---- documents --------------------------------------------------------------------------


def load_documents(storage, unit: str, passage_words: int) -> dict:
    """Documents (texts or passages of texts) and the pages (files) they come from."""
    docs, origin, pages = [], [], []
    skipped = {"unreadable": 0, "empty": 0}
    for entry, content in storage.texts():
        if content is None:
            skipped["unreadable"] += 1
            continue
        if not content.strip():
            skipped["empty"] += 1
            continue
        page_no = len(pages)
        pages.append({"rid": entry.get("@rid"), "label": entry.get("label")})
        if unit == "texts":
            docs.append(content)
            origin.append({"page": page_no, "start_char": 0, "end_char": len(content)})
            continue
        found, starts, ends = words_with_spans(content)
        for first in range(0, len(found), passage_words):
            last = min(first + passage_words, len(found)) - 1
            # a short tail joins the passage before it
            if first and len(found) - first < passage_words // 2:
                origin[-1]["end_char"] = ends[-1]
                docs[-1] = content[origin[-1]["start_char"]:ends[-1]]
                break
            docs.append(content[starts[first]:ends[last]])
            origin.append({"page": page_no, "start_char": starts[first], "end_char": ends[last]})
    return {"docs": docs, "origin": origin, "pages": pages, "skipped": skipped}


def bag_of_words(docs: list[str], stops: set[str], min_docs: int, max_share: float):
    from gensim import corpora

    tokenized = [[w for w in words(d) if w not in stops] for d in docs]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=min_docs, no_above=max_share, keep_n=50000)
    if len(dictionary) < 2:
        raise HTTPException(status_code=400, detail="Too few words left after removing stop words, rare and common words; "
                                                    "lower 'Rare words' or raise 'Common words'")
    corpus = [dictionary.doc2bow(t) for t in tokenized]
    return tokenized, dictionary, corpus


# ---- methods ----------------------------------------------------------------------------
# Each returns (topics, shares): topics as [(topic id, [(word, weight), ...])], shares as one
# {topic id: share} per document (shares of a document sum to 1, or it is empty).


def normalize_shares(pairs) -> dict[int, float]:
    positive = {int(t): float(w) for t, w in pairs if float(w) > 0}
    total = sum(positive.values())
    return {t: w / total for t, w in positive.items()} if total else {}


def fit_lda(corpus, dictionary, k: int, params: dict, seed: int):
    from gensim.models import LdaModel

    passes = int_param(params, "passes", 10, 1, 200)
    model = LdaModel(corpus, id2word=dictionary, num_topics=k, passes=passes, iterations=200,
                     alpha="auto", eta="auto", random_state=seed, eval_every=None)
    topics = [(t, model.show_topic(t, topn=TOP_WORDS)) for t in range(k)]
    shares = [normalize_shares(model.get_document_topics(bow, minimum_probability=0.0)) for bow in corpus]
    return topics, shares


def fit_nmf(corpus, dictionary, k: int, params: dict, seed: int):
    from gensim.models import Nmf, TfidfModel

    passes = int_param(params, "passes", 10, 1, 200)
    tfidf = TfidfModel(corpus)
    weighted = list(tfidf[corpus])
    model = Nmf(weighted, id2word=dictionary, num_topics=k, passes=passes, random_state=seed)
    topics = [(t, model.show_topic(t, topn=TOP_WORDS)) for t in range(k)]
    shares = [normalize_shares(model.get_document_topics(bow, minimum_probability=0.0)) for bow in weighted]
    return topics, shares


def fit_lsi(corpus, dictionary, k: int, params: dict, seed: int):
    from gensim.models import LsiModel, TfidfModel

    tfidf = TfidfModel(corpus)
    weighted = list(tfidf[corpus])
    model = LsiModel(weighted, id2word=dictionary, num_topics=k, random_seed=seed)
    k = model.num_topics  # fewer when the matrix has a lower rank
    topics, signs = [], {}
    for t in range(k):
        # LSI topics weigh words both ways, and the sign of an SVD direction is arbitrary: turn
        # each topic so that its strongest words weigh positively, and name it by those
        found = model.show_topic(t, topn=len(dictionary))
        signs[t] = -1.0 if sum(w for _, w in found[:TOP_WORDS]) < 0 else 1.0
        topics.append((t, sorted(((word, signs[t] * w) for word, w in found), key=lambda x: -x[1])[:TOP_WORDS]))
    shares = [normalize_shares((t, signs.get(t, 1.0) * w) for t, w in model[bow]) for bow in weighted]
    return topics, shares


def fit_hdp(corpus, dictionary, k: int, params: dict, seed: int):
    from gensim.models import HdpModel

    passes = int_param(params, "passes", 10, 1, 200)
    # gensim's HDP is online: one pass over a small corpus leaves it barely trained
    model = HdpModel(corpus, id2word=dictionary, chunksize=min(256, max(16, len(corpus) // 8)), random_state=seed)
    for _ in range(passes - 1):
        model.update(corpus)
    shares = [normalize_shares(model[bow]) for bow in corpus]
    # HDP keeps many small topics; those that hold at least 10 % of some document are shown
    used = sorted({t for s in shares for t, w in s.items() if w >= HDP_MIN_SHARE})
    shares = [normalize_shares((t, w) for t, w in s.items() if t in used) for s in shares]
    topics = [(t, model.show_topic(t, topn=TOP_WORDS)) for t in used]
    return topics, shares


def fit_kmeans(corpus, dictionary, k: int, params: dict, seed: int):
    from gensim.matutils import corpus2csc
    from gensim.models import TfidfModel
    from sklearn.cluster import KMeans

    tfidf = TfidfModel(corpus)
    matrix = corpus2csc(tfidf[corpus], num_terms=len(dictionary)).T.tocsr()  # documents x words, L2-normalised
    model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(matrix)
    topics = []
    for t, centre in enumerate(model.cluster_centers_):
        top = np.argsort(-centre)[:TOP_WORDS]
        topics.append((t, [(dictionary[int(i)], float(centre[i])) for i in top if centre[i] > 0]))
    shares = [{int(label): 1.0} for label in model.labels_]
    extra = {}
    if 2 <= k < matrix.shape[0]:
        from sklearn.metrics import silhouette_score

        extra["silhouette"] = round(float(silhouette_score(matrix, model.labels_, metric="cosine",
                                                           sample_size=min(2000, matrix.shape[0]), random_state=seed)), 4)
    return topics, shares, extra


FITTERS = {"lda": fit_lda, "nmf": fit_nmf, "lsi": fit_lsi, "hdp": fit_hdp, "kmeans": fit_kmeans}


def coherence(topics, tokenized, dictionary) -> tuple[float | None, list[float | None]]:
    """c_v coherence of each topic's top words (and their mean), on at most COHERENCE_MAX_DOCS texts."""
    from gensim.models import CoherenceModel

    word_lists = [[w for w, _ in found if w in dictionary.token2id] for _, found in topics]
    scored = [i for i, wl in enumerate(word_lists) if len(wl) >= 2]
    if not scored:
        return None, [None] * len(topics)
    texts = tokenized
    if len(texts) > COHERENCE_MAX_DOCS:
        step = len(texts) / COHERENCE_MAX_DOCS
        texts = [texts[int(i * step)] for i in range(COHERENCE_MAX_DOCS)]
    try:
        model = CoherenceModel(topics=[word_lists[i] for i in scored], texts=texts, dictionary=dictionary,
                               coherence="c_v", processes=1)
        per_topic = model.get_coherence_per_topic()
    except Exception:
        logger.exception("coherence failed (the topics are still written)")
        return None, [None] * len(topics)
    out: list[float | None] = [None] * len(topics)
    for i, score in zip(scored, per_topic):
        out[i] = round(float(score), 4) if np.isfinite(score) else None
    finite = [s for s in out if s is not None]
    return (round(float(np.mean(finite)), 4) if finite else None), out


# ---- output -----------------------------------------------------------------------------


def topic_label(found: list[tuple[str, float]]) -> str:
    top = [w for w, _ in found[:3] if w]
    return " · ".join(top) if top else "(no words)"


def describe(topics, shares, corpus: dict, per_topic_coherence) -> dict:
    origin, pages = corpus["origin"], corpus["pages"]
    labels = {t: topic_label(found) for t, found in topics}
    main = [max(s, key=s.get) if s else None for s in shares]
    out_topics = []
    for (tid, found), score in zip(topics, per_topic_coherence):
        members = [i for i, m in enumerate(main) if m == tid]
        closest = sorted((i for i, s in enumerate(shares) if s.get(tid, 0) > 0), key=lambda i: -shares[i][tid])[:3]
        out_topics.append({
            "id": int(tid),
            "label": labels[tid],
            "outliers": False,
            "size": len(members),
            "coherence": score,
            "words": [{"word": w, "weight": round(float(s), 4)} for w, s in found],
            "representative": [{"rid": pages[origin[i]["page"]]["rid"], "label": pages[origin[i]["page"]]["label"],
                                "start_char": origin[i]["start_char"], "end_char": origin[i]["end_char"],
                                "share": round(shares[i][tid], 3)} for i in closest],
        })

    # a page's share of a topic: the mean over its documents (one document per page with unit texts)
    totals: list[dict[int, float]] = [dict() for _ in pages]
    counts = [0] * len(pages)
    for i, doc_shares in enumerate(shares):
        page = origin[i]["page"]
        counts[page] += 1
        for tid, s in doc_shares.items():
            totals[page][tid] = totals[page].get(tid, 0.0) + s
    out_pages, file_tags = [], {}
    for page, sums, count in zip(pages, totals, counts):
        if not sums:
            continue
        ranked = sorted(((tid, s / count) for tid, s in sums.items()), key=lambda x: -x[1])
        top = ranked[0][0]
        out_pages.append({**page, "topic": int(top), "topic_label": labels.get(top, ""),
                          "shares": [{"topic": int(tid), "share": round(s, 3)} for tid, s in ranked if s >= 0.005]})
        tags = [{"label": labels[tid], "confidence": round(s, 3)} for tid, s in ranked if s >= TAG_MIN_SHARE][:TAG_MAX_PER_PAGE]
        if tags and page.get("rid"):
            file_tags[page["rid"]] = tags
    return {"topics": out_topics, "pages": out_pages, "file_tags": file_tags}


def overview_png(path, result: dict, title: str) -> None:
    """The top words of each topic as small bar charts (at most 20 topics, largest first)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = sorted(result["topics"], key=lambda t: -t["size"])[:20]
    cols = min(4, max(1, len(shown)))
    rows = (len(shown) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows + 0.8), squeeze=False)
    fig.suptitle(title, fontsize=13)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, topic in zip(axes.flat, shown):
        ax.set_visible(True)
        found = topic["words"][::-1]
        ax.barh([w["word"] for w in found], [w["weight"] for w in found], color="#0f6e6a")
        heading = f'{topic["id"]}: {topic["size"]} docs'
        if topic.get("coherence") is not None:
            heading += f' · c_v {topic["coherence"]:.2f}'
        ax.set_title(heading, fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_topics(message: dict, storage) -> dict:
    task = message.get("task") or {}
    params = task.get("params") or {}
    method = str(params.get("method") or "lda").strip().lower()
    if method not in FITTERS:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method} (one of {', '.join(FITTERS)})")
    one_text = not message.get("files")
    unit = "passages" if one_text or str(params.get("unit") or "texts").lower().startswith("passage") else "texts"
    passage_words = int_param(params, "passage_words", 200, 20, 10000)
    seed = int_param(params, "seed", 42, 0)
    languages = languages_from(params)
    stops = stop_words(languages, str(params.get("extra_stopwords") or ""))

    corpus = load_documents(storage, unit, passage_words)
    n = len(corpus["docs"])
    if n < MIN_DOCUMENTS:
        raise HTTPException(status_code=400, detail=f"Topic modelling needs at least {MIN_DOCUMENTS} {unit}; this input has {n}")
    k = int_param(params, "num_topics", 10, 1, 500)
    if method == "kmeans":
        k = min(k, n)
    min_docs = int_param(params, "min_docs", 2 if n >= 10 else 1, 1)
    max_share = int_param(params, "max_share", 50, 1, 100) / 100
    if n < 10:
        max_share = 1.0  # with a handful of documents almost every word is in "most" of them

    tokenized, dictionary, bow = bag_of_words(corpus["docs"], stops, min_docs, max_share)
    logger.info("fitting %s on %d %s, %d words, k=%d", method, n, unit, len(dictionary), k)
    fitted = FITTERS[method](bow, dictionary, k, params, seed)
    topics, shares = fitted[0], fitted[1]
    extra = fitted[2] if len(fitted) > 2 else {}
    mean_coherence, per_topic = coherence(topics, tokenized, dictionary)
    result = describe(topics, shares, corpus, per_topic)

    out = {
        "format": TOPICS_FORMAT,
        "model": {"service": "md-gensim", "id": method, "name": METHODS[method]},
        "unit": unit,
        "params": {
            "method": method, "num_topics": None if method == "hdp" else k, "seed": seed,
            "passage_words": passage_words if unit == "passages" else None,
            "min_docs": min_docs, "max_share": max_share, "languages": languages,
            "extra_stopwords": str(params.get("extra_stopwords") or ""),
            "passes": int_param(params, "passes", 10, 1, 200) if method in ("lda", "nmf", "hdp") else None,
        },
        "documents": n,
        "vocabulary": len(dictionary),
        "skipped": corpus["skipped"],
        "coherence": {"measure": "c_v", "score": mean_coherence},
        **extra,
        "created": now(),
        **result,
    }
    base = uuid4().hex
    prefix = "" if not one_text else f"{stem((message.get('file') or {}).get('label'))}."
    json_name, png_name = f"{base}.topics.json", f"{base}.png"
    storage.output(json_name).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    title = f"{METHODS[method]} · {n} {unit} · {len(result['topics'])} topics"
    if mean_coherence is not None:
        title += f" · c_v {mean_coherence:.2f}"
    if "silhouette" in extra:
        title += f" · silhouette {extra['silhouette']:.2f}"
    try:
        overview_png(storage.output(png_name), result, title)
        pictures = [{"name": png_name, "label": f"{prefix}topics_{method}.png", "type": "image", "extension": "png"}]
    except Exception:
        logger.exception("topic chart failed (topics.json is still written)")
        pictures = []
    task_id = task.get("id") or "topics"
    return storage.respond(task_id, [{"name": json_name, "label": f"{prefix}topics_{method}.json", "type": "topics.json", "extension": "json"}, *pictures])
