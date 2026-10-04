"""Bag of words: every word of a text with its count, most frequent first."""

import json
from collections import Counter
from uuid import uuid4

from fastapi import HTTPException

from text import bool_param, languages_from, stem, stop_words, words


def run_bow(message: dict, storage) -> dict:
    params = (message.get("task") or {}).get("params") or {}
    content = storage.input().read_bytes().decode("utf-8", errors="replace")
    found = words(content)
    if not found:
        raise HTTPException(status_code=400, detail="No words found in the text")
    if bool_param(params, "remove_stopwords"):
        stops = stop_words(languages_from(params), str(params.get("extra_stopwords") or ""))
        found = [w for w in found if w not in stops]
    counts = [{"word": w, "count": c} for w, c in Counter(found).most_common()]
    name = f"{uuid4().hex}.bow.json"
    storage.output(name).write_text(json.dumps(counts, ensure_ascii=False, indent=1), encoding="utf-8")
    label = f"{stem((message.get('file') or {}).get('label'))}.bow.json"
    return storage.respond("bow", [{"name": name, "label": label, "type": "bow.json", "extension": "json"}])
