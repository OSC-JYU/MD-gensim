"""bow, similarity index + search and topics on small texts, in disk and http mode."""

import json
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bow  # noqa: E402
import similarity  # noqa: E402
import storage  # noqa: E402
import topics  # noqa: E402

THEMES = {
    "sea": "ship harbour sailor anchor wave captain cargo deck storm mast sail port voyage crew",
    "kitchen": "bread oven butter flour recipe cook soup kitchen salt dough bake pepper onion garlic",
    "football": "goal match player referee stadium ball team score league coach pitch kick fans striker",
}


def theme_text(theme: str, n: int, offset: int) -> str:
    vocab = THEMES[theme].split()
    words = [vocab[(offset + i * 3) % len(vocab)] for i in range(n)]
    return " ".join(words).capitalize() + "."


QUOTE = ("The old captain stood on the deck while the storm tore at the mast, and the crew "
         "lowered the sail before the wave reached the harbour wall.")


@pytest.fixture
def md(tmp_path, monkeypatch):
    monkeypatch.setenv("MD_PATH", str(tmp_path))
    monkeypatch.delenv("STORAGE_MODE", raising=False)
    (tmp_path / "data/messydesk/projects").mkdir(parents=True)
    return tmp_path


def write(md, name, content):
    rel = f"data/messydesk/projects/{name}"
    (md / rel).write_text(content, encoding="utf-8")
    return rel


def corpus(md):
    entries = []
    for i, theme in enumerate(["sea", "kitchen", "football"] * 4):
        text = theme_text(theme, 60, i)
        if i == 3:
            text += " " + QUOTE + " " + theme_text(theme, 30, 99)
        rel = write(md, f"text_{i}.txt", text)
        entries.append({"@rid": f"#10:{i}", "label": f"text_{i}.txt", "path": rel, "type": "text", "extension": "txt"})
    return entries


def output_of(md, response, i=0):
    item = response["response"]["files"][i]
    return md / "data/messydesk/tmp" / item["path"], item


def test_windows_cover_the_text_without_nested_tails():
    assert similarity.windows(0, 15, 5) == []
    assert similarity.windows(10, 15, 5) == [0]
    assert similarity.windows(25, 15, 5) == [0, 10]  # 20 + 5 = 25: the tail would be inside [10, 25)
    assert similarity.windows(26, 15, 5) == [0, 10, 20]


def test_bow_counts_words(md):
    rel = write(md, "a.txt", "Ship, ship and the SHIP; harbour and 2024 sea.")
    msg = {"task": {"id": "bow"}, "file": {"@rid": "#1:1", "label": "a.txt", "path": rel}}
    path, item = output_of(md, bow.run_bow(msg, storage.storage_for(msg)))
    counts = json.loads(path.read_text())
    assert counts[0] == {"word": "ship", "count": 3}
    assert {"word": "and", "count": 2} in counts
    assert item["label"] == "a.bow.json" and item["type"] == "bow.json"

    msg["task"]["params"] = {"remove_stopwords": True, "languages": ["en"]}
    path, _ = output_of(md, bow.run_bow(msg, storage.storage_for(msg)))
    assert "and" not in {c["word"] for c in json.loads(path.read_text())}


def index_set(md, entries, **params):
    set_node = {"@rid": "#20:1", "path": "data/messydesk/projects/set_1"}
    msg = {"task": {"id": "similarity_index_set", "params": params}, "file": set_node, "files": entries}
    path, item = output_of(md, similarity.build_index(msg, storage.storage_for(msg)))
    return path, item


def search(md, index_path, query, role="semantic_search", **params):
    rel = str(index_path.relative_to(md))
    msg = {"task": {"id": "search", "params": {"query": query, **params}},
           "file": {"@rid": "#30:1", "path": rel}, "role": role}
    return similarity.search(msg, storage.storage_for(msg))


def test_index_of_a_set_and_a_short_query(md):
    entries = corpus(md)
    entries.append({"@rid": "#10:99", "label": "scan.jpg", "path": "nope.jpg", "type": "image", "extension": "jpg"})
    path, item = index_set(md, entries)
    assert item["type"] == "similarity_index" and item["extension"] == "safetensors"
    assert item["label"] == "tfidf_index_12_texts.safetensors"

    results = search(md, path, "stood tore lowered")["response"]["results"]
    assert results["query_windows"] == 1
    hits = results["chunk_similarities"]
    assert hits and results["doc_map"][hits[0]["doc_index"]] == "#10:3"
    text = (md / entries[3]["path"]).read_text()
    span = text[hits[0]["text_start_char"]:hits[0]["text_end_char"]].lower()
    assert span.startswith(("stood", "tore", "lowered")) and span.endswith(("stood", "tore", "lowered"))
    # overlapping passages of the same text are not listed twice
    spans = [(h["doc_index"], h["chunk"]) for h in hits]
    assert len(spans) == len(set(spans))
    assert all(h["similarity"] <= hits[0]["similarity"] for h in hits)


def test_text_reuse_finds_the_quoted_passage(md):
    entries = corpus(md)
    path, _ = index_set(md, entries, window="10", overlap="3")
    pasted = ("My own words come first here, nothing to see. " + QUOTE.replace("old", "grey") +
              " Then more words of mine that are nowhere else at all.")
    results = search(md, path, pasted)["response"]["results"]
    assert results["query_windows"] > 1 and results["threshold"] == 0.3
    matches = results["chunk_similarities"]
    assert matches and all(results["doc_map"][m["doc_index"]] == "#10:3" for m in matches)
    text = (md / entries[3]["path"]).read_text()
    best = max(matches, key=lambda m: m["similarity"])
    assert best["similarity"] > 0.8
    assert "captain" in text[best["text_start_char"]:best["text_end_char"]] or "sail" in text[best["text_start_char"]:best["text_end_char"]]
    # both sides are narrowed to the shared words, so the spans hold the same words
    query_words = pasted[best["query_start_char"]:best["query_end_char"]].lower().split()
    text_words = text[best["text_start_char"]:best["text_end_char"]].lower().split()
    assert query_words[-1].strip(",.") == text_words[-1].strip(",.")
    # query_start_token counts whitespace-separated words, as the UI does
    assert pasted.split()[best["query_start_token"]].lower().strip(",.") == query_words[0].strip(",.")
    # my own words before and after the quote match nothing
    assert all(m["query_start_char"] >= pasted.index("The grey") for m in matches if m["similarity"] > 0.5)


def test_search_as_a_job_writes_similarity_json(md):
    path, _ = index_set(md, corpus(md))
    response = search(md, path, "bread oven butter", role=None)
    out, item = output_of(md, response)
    assert item["type"] == "similarity.json"
    assert json.loads(out.read_text())["chunk_similarities"]


def test_index_of_one_text(md):
    rel = write(md, "book.txt", theme_text("sea", 200, 0) + " " + QUOTE)
    msg = {"task": {"id": "similarity_index"}, "file": {"@rid": "#1:5", "label": "book.txt", "path": rel}}
    path, item = output_of(md, similarity.build_index(msg, storage.storage_for(msg)))
    assert item["label"] == "book.tfidf_index.safetensors"
    results = search(md, path, "lowered the sail before the wave")["response"]["results"]
    assert results["doc_map"] == ["#1:5"]


def test_empty_query_and_unknown_words(md):
    path, _ = index_set(md, corpus(md))
    with pytest.raises(HTTPException):
        search(md, path, "   ")
    assert search(md, path, "zzzz qqqq")["response"]["results"]["chunk_similarities"] == []


def test_http_mode_reads_the_set_from_a_zip(md, tmp_path, monkeypatch):
    entries = corpus(md)
    monkeypatch.setenv("STORAGE_MODE", "http")
    archive = tmp_path / "set.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for e in entries:
            z.write(md / e["path"], e["label"])
    store = tmp_path / "store"
    store.mkdir()
    msg = {"task": {"id": "similarity_index_set"}, "file": {"@rid": "#20:1"}, "files": entries}
    response = similarity.build_index(msg, storage.storage_for(msg, archive, store))
    item = response["response"]["uri"][0]
    assert item["uri"].startswith("/files/") and item["label"] == "tfidf_index_12_texts"
    index_file = store / item["uri"].rsplit("/", 1)[1]
    query = {"task": {"id": "search", "params": {"query": "goal referee stadium"}}, "file": {"@rid": "#30:1"}, "role": "semantic_search"}
    results = similarity.search(query, storage.storage_for(query, index_file, store))["response"]["results"]
    assert results["doc_map"][results["chunk_similarities"][0]["doc_index"]] in {f"#10:{i}" for i in (2, 5, 8, 11)}


def run_topics(md, entries, **params):
    set_node = {"@rid": "#20:1", "path": "data/messydesk/projects/set_1"}
    msg = {"task": {"id": "topics", "params": params}, "file": set_node, "files": entries}
    response = topics.run_topics(msg, storage.storage_for(msg))
    path, item = output_of(md, response)
    return json.loads(path.read_text()), response


@pytest.mark.parametrize("method", ["lda", "nmf", "lsi", "hdp", "kmeans"])
def test_topic_methods(md, method):
    entries = corpus(md)
    out, response = run_topics(md, entries, method=method, num_topics="3", languages=["en"], autotag=True)
    files = response["response"]["files"]
    assert files[0]["type"] == "topics.json" and files[0]["label"] == f"topics_{method}.json"
    assert files[1]["type"] == "image" and files[1]["extension"] == "png"
    assert out["format"] == "messydesk-topics/1" and out["model"]["id"] == method
    assert out["documents"] == 12 and out["unit"] == "texts"
    assert out["topics"] and all(t["words"] for t in out["topics"])
    assert len(out["pages"]) == 12
    for page in out["pages"]:
        assert abs(sum(s["share"] for s in page["shares"]) - 1) < 0.05
    assert "coherence" in out and out["coherence"]["measure"] == "c_v"
    if method != "hdp":
        assert len(out["topics"]) <= 3
    if method == "kmeans":
        assert "silhouette" in out
        # the three themes are three clusters
        by_theme = {}
        for page in out["pages"]:
            by_theme.setdefault(int(page["rid"].split(":")[1]) % 3, set()).add(page["topic"])
        assert all(len(t) == 1 for t in by_theme.values())
        assert len({next(iter(t)) for t in by_theme.values()}) == 3
    if method in ("lda", "nmf", "kmeans"):
        assert out["file_tags"] and set(out["file_tags"]) <= {e["@rid"] for e in entries}


def test_topics_in_passages_of_one_text(md):
    text = " ".join(theme_text(t, 120, i) for i, t in enumerate(["sea", "kitchen", "football", "sea", "kitchen", "football"]))
    rel = write(md, "long.txt", text)
    msg = {"task": {"id": "topics_text", "params": {"method": "nmf", "num_topics": "3", "passage_words": "60"}},
           "file": {"@rid": "#1:7", "label": "long.txt", "path": rel}}
    response = topics.run_topics(msg, storage.storage_for(msg))
    path, item = output_of(md, response)
    out = json.loads(path.read_text())
    assert item["label"] == "long.topics_nmf.json"
    assert out["unit"] == "passages" and out["documents"] == 12
    rep = out["topics"][0]["representative"][0]
    assert rep["rid"] == "#1:7" and 0 <= rep["start_char"] < rep["end_char"] <= len(text)


def test_topics_parameter_errors(md):
    entries = corpus(md)
    with pytest.raises(HTTPException):
        run_topics(md, entries, method="word2vec")
    with pytest.raises(HTTPException):
        run_topics(md, entries[:1])
    with pytest.raises(HTTPException):
        run_topics(md, entries, num_topics="many")
