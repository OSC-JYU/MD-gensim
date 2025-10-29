
# MD-gensim

An experimental MessyDesk wrapper for gensim NLP librarary


## API

endpoint is http://localhost:9009/process

Payload is queue message as json file. 

## Running as service (locally)

Create .env file with MD_PATH like this:

	MD_PATH="/home/you/Projects/MessyDesk"

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



