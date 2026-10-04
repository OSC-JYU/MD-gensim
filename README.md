# MD-gensim

Classic text analysis for [MessyDesk](https://github.com/OSC-JYU/MessyDesk) with
[Gensim](https://radimrehurek.com/gensim/): word counts, TF-IDF similarity (search and text reuse)
and topic models. The user help is [index.md](index.md) (served at `/help`); the tasks and their
settings are in [service.json](service.json) (served at `/config`).

| Task | Input | Output |
|---|---|---|
| `bow` | a text | `<name>.bow.json`: `[{word, count}]`, most frequent first |
| `similarity_index` | a text | `similarity_index` file (`.safetensors`) |
| `similarity_index_set` | a set (whole-set) | `similarity_index` file of all its texts |
| `search` (internal) | a `similarity_index` + `params.query` | hits, see below |
| `topics` | a set (whole-set) | `topics_<method>.json` (`topics.json`) and a chart (`.png`) |
| `topics_text` | a text | the same, for the passages of the text |

## Similarity

`similarity_index*` stores the words of every text as ids into a vocabulary, with the character
span of each word, in a safetensors file (format `messydesk-tfidf-index/1`; vocabulary, files and
settings in the header's `__metadata__`, like MD-embeddings' vector index). The passages (`window`
words, `overlap` shared), the TF-IDF model and a gensim `SparseMatrixSimilarity` are rebuilt when
the index is first searched and kept in memory (the last four indexes).

`search` takes `params.query`, `top_k` (default 20) and `threshold` (default 0.3). The query is
split into passages of the index's length:

- one passage (a short query): the `top_k` best passages, without overlapping ones;
- several (a pasted text): for each query passage the best passage of the index, if at least
  `threshold` alike.

The result has the same shape as MD-embeddings' search (`doc_map`, `chunk_similarities` with
`doc_index`, `similarity`, `text_start_char`, `text_end_char`), plus `query_start_char`,
`query_end_char` and `query_start_token` (in whitespace-separated words, as the UI counts them).
Both spans are narrowed to the first and last word the two passages share. With
`role: "semantic_search"` (the backend's interactive search) the results come back in the answer;
otherwise they are written to a `similarity.json` file.

## Topics

`params.method` is `lda`, `nmf`, `lsi`, `hdp` or `kmeans` (scikit-learn); see [index.md](index.md)
for what each does. The output follows MD-bertopic's `messydesk-topics/1`: `topics` (words with
weights, size, representative documents, c_v `coherence`), `pages` (each text's main topic and
shares) and `file_tags` for autotagging, plus the run's mean `coherence` and, for k-means,
`silhouette`.

## Running

```bash
make build
make start
```

The service listens on port 9009. Without podman, `CONTAINER_RUNTIME=docker make build`.

Storage modes, as in MD-embeddings and MD-bertopic:

- **disk** (`elg_fs` adapter) when `MD_PATH` is set: inputs are read under `MD_PATH`, outputs are
  written to `MD_PATH/data/<db>/tmp`. Mount MessyDesk's data and set `MD_PATH` to the mount's
  parent, e.g. `-v /path/to/MessyDesk/data:/md/data -e MD_PATH=/md`.
- **http** (`elg` adapter) otherwise, or with `STORAGE_MODE=http`: the input is the `content`
  upload (a set as a zip of its files by label), outputs are served from `/files`.

`/config` reports the adapter that matches the mode.

### Example call (disk mode)

```bash
curl -F 'message={"task":{"id":"bow"},"file":{"@rid":"#1:1","label":"a.txt","path":"data/messydesk/projects/a.txt"}};type=application/json' \
  http://localhost:9009/process
```

## Tests

```bash
make test
```

Runs the unit tests in the service image (gensim has no wheels for the newest Python versions).
