"""TF-IDF similarity: an index of one text or a set, and searches of it with a pasted text.

The index is a safetensors file, like MD-embeddings' vector index, so the backend reads its header
the same way. It keeps the words of every text (as ids into the vocabulary) and where each word is
in the text; the passages, TF-IDF weights and gensim's similarity matrix are rebuilt from them when
the index is first searched, and kept in memory for the next searches.

A search splits the query into passages of the same length. A short query (one passage) gets the
best passages of the index, as in the Search tab; a longer text gets, for each of its passages, the
passage of the index it is most like, if it is at least `threshold` alike (text reuse).
"""

import bisect
import json
import logging
import re
import threading
from collections import Counter, OrderedDict
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import HTTPException
from safetensors import safe_open
from safetensors.numpy import save_file

from text import int_param, float_param, now, stem, words_with_spans

logger = logging.getLogger("md-gensim")

INDEX_FORMAT = "messydesk-tfidf-index/1"
INDEX_TYPE = "similarity_index"
DEFAULT_WINDOW = 15
DEFAULT_OVERLAP = 5
DEFAULT_THRESHOLD = 0.3
MAX_TOP_K = 500
CACHE_SIZE = 4


def windows(count: int, size: int, overlap: int) -> list[int]:
    """Start positions of passages of `size` words, `overlap` words shared with the previous one.
    The last passage may be shorter, but is never contained in the one before it."""
    step = max(1, size - overlap)
    return [start for start in range(0, count, step) if start == 0 or start + overlap < count]


# ---- index ------------------------------------------------------------------------------


def build_index(message: dict, storage) -> dict:
    params = (message.get("task") or {}).get("params") or {}
    size = int_param(params, "window", DEFAULT_WINDOW, 3, 500)
    overlap = int_param(params, "overlap", DEFAULT_OVERLAP, 0, size - 1)

    vocab: dict[str, int] = {}
    token_ids: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    doc_offsets = [0]
    files: list[dict] = []
    skipped = 0
    for entry, content in storage.texts():
        found, word_starts, word_ends = words_with_spans(content or "")
        if not found:
            skipped += 1
            continue
        token_ids.extend(vocab.setdefault(w, len(vocab)) for w in found)
        starts.extend(word_starts)
        ends.extend(word_ends)
        doc_offsets.append(len(token_ids))
        files.append({"rid": entry.get("@rid"), "label": entry.get("label"), "chars": len(content)})
    if not files:
        raise HTTPException(status_code=400, detail="No words found in the input texts")

    rows = sum(len(windows(doc_offsets[i + 1] - doc_offsets[i], size, overlap)) for i in range(len(files)))
    model = {"service": "md-gensim", "id": "tf-idf", "name": "TF-IDF (gensim)", "window": size, "overlap": overlap}
    tensors = {
        "token_ids": np.asarray(token_ids, dtype=np.int32),
        "token_start": np.asarray(starts, dtype=np.int32),
        "token_end": np.asarray(ends, dtype=np.int32),
        "doc_offsets": np.asarray(doc_offsets, dtype=np.int32),
    }
    vocabulary = sorted(vocab, key=vocab.get)
    metadata = {
        "format": INDEX_FORMAT,
        "model": json.dumps(model),
        "vocabulary": json.dumps(vocabulary, ensure_ascii=False),
        "files": json.dumps(files, ensure_ascii=False),
        "rows": str(rows),
        "words": str(len(token_ids)),
        "skipped": str(skipped),
        "created": now(),
    }
    name = f"{uuid4().hex}.safetensors"
    target = storage.output(name)
    save_file(tensors, str(target), metadata=metadata)
    target.chmod(0o644)  # save_file creates 0600; the backend may run as another user
    task = (message.get("task") or {}).get("id") or "similarity_index"
    if message.get("files"):
        label = f"tfidf_index_{len(files)}_texts.safetensors"
    else:
        label = f"{stem((message.get('file') or {}).get('label'))}.tfidf_index.safetensors"
    return storage.respond(task, [{"name": name, "label": label, "type": INDEX_TYPE, "extension": "safetensors"}])


# ---- loading ----------------------------------------------------------------------------


class LoadedIndex:
    """An index file with its passages, TF-IDF model and gensim similarity matrix."""

    def __init__(self, path: Path):
        from gensim import models, similarities

        with safe_open(str(path), framework="np") as f:
            meta = f.metadata() or {}
            if meta.get("format") != INDEX_FORMAT:
                raise HTTPException(status_code=400, detail="Not a MessyDesk TF-IDF index")
            self.token_ids = f.get_tensor("token_ids")
            self.token_start = f.get_tensor("token_start")
            self.token_end = f.get_tensor("token_end")
            doc_offsets = f.get_tensor("doc_offsets")
        self.model = json.loads(meta["model"])
        self.files = json.loads(meta["files"])
        vocabulary = json.loads(meta["vocabulary"])
        self.vocab = {w: i for i, w in enumerate(vocabulary)}
        self.size = int(self.model["window"])
        self.overlap = int(self.model["overlap"])

        # passages: (document, first word, end word) in the flat word arrays
        self.passages: list[tuple[int, int, int]] = []
        for doc in range(len(doc_offsets) - 1):
            first, last = int(doc_offsets[doc]), int(doc_offsets[doc + 1])
            for start in windows(last - first, self.size, self.overlap):
                self.passages.append((doc, first + start, min(first + start + self.size, last)))
        corpus = [sorted(Counter(self.token_ids[a:b].tolist()).items()) for _, a, b in self.passages]
        self.tfidf = models.TfidfModel(corpus, normalize=True)
        self.index = similarities.SparseMatrixSimilarity(self.tfidf[corpus], num_features=len(vocabulary))

    def bow(self, found: list[str]) -> list[tuple[int, int]]:
        return sorted(Counter(self.vocab[w] for w in found if w in self.vocab).items())

    def scores(self, found: list[str]) -> np.ndarray | None:
        bow = self.bow(found)
        if not bow:
            return None
        return np.asarray(self.index[self.tfidf[bow]], dtype=np.float32).reshape(-1)

    def match_span(self, passage: int, found: list[str]) -> tuple[int, int]:
        """Characters from the first to the last word of the passage that is also in the query."""
        wanted = {self.vocab[w] for w in found if w in self.vocab}
        _, a, b = self.passages[passage]
        shared = [i for i in range(a, b) if int(self.token_ids[i]) in wanted] or [a, b - 1]
        return int(self.token_start[shared[0]]), int(self.token_end[shared[-1]])


_cache: "OrderedDict[tuple[str, float], LoadedIndex]" = OrderedDict()
_cache_lock = threading.Lock()


def load_index(path: Path) -> LoadedIndex:
    key = (str(path), path.stat().st_mtime)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    loaded = LoadedIndex(path)
    with _cache_lock:
        _cache[key] = loaded
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return loaded


# ---- search -----------------------------------------------------------------------------


def search(message: dict, storage) -> dict:
    task = message.get("task") or {}
    params = task.get("params") or {}
    query = str(params.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is empty")
    k = int_param(params, "top_k", 20, 1, MAX_TOP_K)
    threshold = float_param(params, "threshold", DEFAULT_THRESHOLD)

    index = load_index(storage.input())
    found, q_starts, q_ends = words_with_spans(query)
    if not found:
        raise HTTPException(status_code=400, detail="The query has no words")
    # the UI counts query words by whitespace, so passage starts are also given in those
    spaced = [m.start() for m in re.finditer(r"\S+", query)]

    def match(passage: int, score: float, q_first: int, q_last: int) -> dict:
        doc, a, b = index.passages[passage]
        start, end = index.match_span(passage, found[q_first:q_last])
        # the query side is narrowed the same way: its first to last word that is in the passage
        in_passage = set(index.token_ids[a:b].tolist())
        shared = [i for i in range(q_first, q_last) if index.vocab.get(found[i], -1) in in_passage]
        if shared:
            q_first, q_last = shared[0], shared[-1] + 1
        info = index.files[doc]
        return {
            "similarity": round(float(score), 4),
            "doc_index": doc,
            "doc_label": info.get("label"),
            "chunk": passage,
            "text_start_char": start,
            "text_end_char": end,
            "query_start_char": q_starts[q_first],
            "query_end_char": q_ends[q_last - 1],
            "query_start_token": max(0, bisect.bisect_right(spaced, q_starts[q_first]) - 1),
        }

    matches: list[dict] = []
    query_windows = windows(len(found), index.size, index.overlap)
    if len(query_windows) == 1:
        # a short query: the best passages, skipping ones that overlap a better one
        scores = index.scores(found)
        if scores is not None:
            taken: list[tuple[int, int, int]] = []
            for passage in np.argsort(-scores):
                score = float(scores[passage])
                if score <= 0 or len(matches) >= k:
                    break
                doc, a, b = index.passages[passage]
                if any(doc == d and a < tb and ta < b for d, ta, tb in taken):
                    continue
                taken.append((doc, a, b))
                matches.append(match(int(passage), score, 0, len(found)))
    else:
        # a text: for each of its passages, the most similar passage of the index
        for q_first in query_windows:
            q_last = min(q_first + index.size, len(found))
            scores = index.scores(found[q_first:q_last])
            if scores is None:
                continue
            best = int(np.argmax(scores))
            if float(scores[best]) >= threshold:
                matches.append(match(best, float(scores[best]), q_first, q_last))

    used = sorted({m["doc_index"] for m in matches})
    remap = {doc: i for i, doc in enumerate(used)}
    for m in matches:
        m["doc_index"] = remap[m["doc_index"]]
    doc_map = [index.files[doc].get("rid") for doc in used]

    out = {
        "query_text": query,
        "model": index.model,
        "index_file": (message.get("file") or {}).get("@rid"),
        "window_size": index.size,
        "overlap": index.overlap,
        "threshold": threshold if len(query_windows) > 1 else None,
        "query_windows": len(query_windows),
        "matched_windows": len(matches) if len(query_windows) > 1 else None,
        "max_similarity": max((m["similarity"] for m in matches), default=None),
        "chunk_count": len(matches),
        "doc_map": doc_map,
        "chunk_similarities": matches,
        "created": now(),
    }
    if message.get("role") == "semantic_search":
        # Interactive search (Search tab, index viewer): the hits go back in the answer, no file.
        return {"task": "search", "response": {"type": "results", "results": out}}
    slug = re.sub(r"[^\w-]+", "_", query.lower())[:40].strip("_") or "query"
    name = f"{uuid4().hex}.similarity.json"
    storage.output(name).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return storage.respond("search", [{"name": name, "label": f"similarity_{slug}.json", "type": "similarity.json", "extension": "json"}])
