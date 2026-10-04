# Gensim

Classic, explainable text analysis with [Gensim](https://radimrehurek.com/gensim/): every result
can be traced back to word counts. Three kinds of tasks:

- **Bag of words**: the words of a text and how often each occurs.
- **Similarity index**: find where a text you paste shares passages with your texts (text reuse,
  quotations, paraphrases), or search the texts by their words.
- **Topics**: find the themes of a set of texts with one of five methods, and compare the methods.

Words are runs of letters, lowercased; numbers and one-letter words are left out.

## Similarity index

1. Run **Similarity index** on a text (for example a book) or on a set of texts.
2. Open the resulting index file and paste another text into it. The result shows which passages
   of your text match which passages of the indexed texts, and how alike they are.

Indexes of sets also appear in the **Search** tab, where you can search them with a few words.

The texts are compared in short **passages** (15 words by default, 5 of them shared with the next
passage). Each passage is a TF-IDF vector: a word weighs more the more often it is in the passage
and the rarer it is in the indexed texts, so names and rare words decide a match more than *and*
or *the*. Two passages are compared by the angle between their vectors (cosine similarity, 0 to 1).

- A **short query** (one passage or less) lists the passages most like it.
- A **longer text** is split into passages too, and each gets the indexed passage most like it,
  when the similarity is at least 0.3. Exact quotes usually score above 0.7 (the passages of the
  two texts rarely start at the same word); reworded passages score lower.

Settings: **Passage length**: short passages find short quotes, long ones find looser paraphrases
but miss short quotes. **Overlap**: words shared by consecutive passages, so a match is not cut in
two.

## Topics

Run **Topics** on a set of texts (each text is a document), or on one long text (it is split into
passages). The result is **topics.json** with each topic's words, its documents and how much of
each text belongs to each topic, and a chart of each topic's words.

Topic models are easiest to understand by trying several. Run the task again with another
**Method** or **Number of topics** and compare the charts and the scores.

| Method | What a topic is | Good to know |
|---|---|---|
| **LDA** | A probability distribution over words; each document is a mix of topics | The classic. Needs a few hundred documents to be stable; more **Passes** help |
| **NMF** | A non-negative part of the TF-IDF matrix; documents are sums of topics | Often clearer topics than LDA on short or noisy (OCR) texts, and faster |
| **LSI** | A direction in word space (singular value decomposition) | Very fast. Topics can weigh words negatively; only the positive side is shown |
| **HDP** | Like LDA, but it chooses the number of topics itself | Use when you have no idea how many topics there are. On varied texts it finds many small topics; more **Passes** help |
| **k-means** | A cluster: each document belongs to exactly one | Not a topic model but clustering; the words are those that weigh most in the cluster's centre |

All methods start from the same **bag of words**: stop words (*and, the, ja, että*) of the chosen
languages are removed, and so are **rare words** (in fewer than 2 documents) and **common words**
(in more than half of the documents). With fewer than 10 documents no common words are removed.

### Comparing runs

- **Coherence (c_v)**, 0 to 1, for every method: how often the top words of a topic occur near
  each other in the texts. Higher usually means topics a person can name. Each topic has its own
  score; the run's score is their mean. Compare runs of the same texts only.
- **Silhouette**, −1 to 1, for k-means: how much closer documents are to their own cluster than
  to the next one. Near 0 means the clusters overlap.

### Settings

- **Unit**: *Texts* makes each text one document; *Passages* splits texts into passages of
  **Passage length** words, so a long text can have several topics.
- **Stop-word languages**, **Other languages** and **Extra stop words**: as in Topics (BERTopic).
- **Tag texts with their topics**: links each text to the topics that cover at least 30 % of it
  (at most three) as machine tags in the **Tags** tab.
- **Random seed**: the same seed and settings give the same result.

## About Gensim

[Gensim](https://radimrehurek.com/gensim/) by Radim Řehůřek and contributors (LGPL-2.1).
k-means and the silhouette score come from [scikit-learn](https://scikit-learn.org/), stop words
from [stopwords-iso](https://github.com/stopwords-iso/stopwords-iso).
