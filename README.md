
# MD-gensim

## Introduction

MD-gensim is a Natural Language Processing (NLP) service for **MessyDesk**, a digital humanities platform. This service provides text analysis capabilities using the Gensim library, enabling researchers and scholars to analyze historical texts, manuscripts, and other digital humanities materials.

### Functionality

The API offers three main services:

1. **Bag of Words (BOW)** - Analyzes text to extract word frequencies, providing a sorted list of all words and their occurrence counts. This helps researchers identify key terms, analyze vocabulary patterns, and understand textual content at a word-level.

2. **Similarity Index Creation** - Builds a searchable similarity index from document texts using TF-IDF (Term Frequency-Inverse Document Frequency) and sliding window chunking. The index is stored as a compressed archive.

3. **Similarity Query** - Searches pre-built similarity indexes to find passages in documents that match a given query text. The service returns similarity scores and accurately maps results back to their original character positions in the source text, enabling precise citation and reference in digital humanities research.

All services preserve original text positions and maintain token-to-character mappings, ensuring that search results can be accurately referenced back to the original source material—a critical requirement for scholarly citation and textual analysis.

## API

endpoint is http://localhost:9009/process

Payload is queue message as json file. 

## Running as service (locally)


Then build and start

	make build
	make start

or start container directly

 	docker run --name md-gensim -p 9009:9009  



### Example API call 

Run these from MD-gensim directory:

Bag of Words:

	curl -X POST -H "Content-Type: multipart/form-data" \
	  -F "message=@test/bow.json;type=application/json" \
	  -F "content=@test/text_fi.txt;type=plain/text" \
	  http://localhost:9009/process


Similarity index creation (httpie version):

	http POST :9009/process message@test/similarity.json content@test/text_fi.txt --form



