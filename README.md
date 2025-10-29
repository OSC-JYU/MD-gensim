
# MD-gensim

## Introduction

MD-gensim is a Natural Language Processing (NLP) service for **MessyDesk**, a digital humanities platform. This service provides text analysis capabilities using the Gensim library.

### Functionality

The API offers three main services:

1. **Bag of Words (BOW)** - Creates a bag of words list as JSON file (word +  word count).

2. **Similarity Index Creation** - Builds a searchable similarity index from document texts using TF-IDF (Term Frequency-Inverse Document Frequency) and sliding window chunking. The index is stored as a compressed archive.

3. **Similarity Query** - Searches pre-built similarity indexes to find passages in documents that match a given query text. 


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



