from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import uuid
import json
from typing import List, Dict, Any, Union
import io
import gensim
from similarity import IncrementalSimilarityBuilder
from pathlib import Path


builder = IncrementalSimilarityBuilder(window_size=15, overlap=5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown


app = FastAPI(
    title="gensim NLP API",
    description="API for gensim NLP tasks",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

OUTPUT_FOLDER = 'output'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.get("/")
async def root():
    return {"message": "gensim NLP API for MessyDesk"}


@app.post("/process")
async def process_files(
    message: UploadFile = File(...), 
    content: UploadFile = File(...),
    source: UploadFile = File(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    if not message or not content:
        raise HTTPException(status_code=400, detail='JSON file and text file are required')
    if message.filename == '' or content.filename == '':
        raise HTTPException(status_code=400, detail='Empty file submitted')
    
    try:
        
        # Parse request JSON
        message_data = await message.read()
        try:
            message_text = message_data.decode('utf-8')
            msg = json.loads(message_text)
            
            # If the parsed result is a string, parse it again to get the actual JSON object
            if isinstance(msg, str):
                msg = json.loads(msg)

            # Read content as bytes, decode only if it's a text file
            print(f"Content type: {content.content_type}")
            print(f"Filename: {content.filename}")
            content_bytes = await content.read()

            try:
                content_text = content_bytes.decode('utf-8')
                content_payload = content_text
            except UnicodeDecodeError:
                # If declared text but not decodable, keep as binary
                content_payload = content_bytes
                
            if source:
                try:
                    source_bytes = await source.read()
                    source_text = source_bytes.decode('utf-8')
                    source_payload = source_text
                    print(source_payload)
                except UnicodeDecodeError:
                    # If declared text but not decodable, keep as binary
                    source_payload = source_bytes

                
        except UnicodeDecodeError as e:
            raise HTTPException(status_code=400, detail=f'Message file encoding error: {str(e)}')
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f'Invalid JSON in message file: {str(e)}')


        # TASKS
        if msg.get("task", {}).get("id") == "bow":
            if isinstance(content_payload, bytes):
                raise HTTPException(status_code=400, detail='BOW expects a text file')
            response = bow(content_payload, msg)

        elif msg.get("task", {}).get("id") == "similarity":
            if isinstance(content_payload, bytes):
                raise HTTPException(status_code=400, detail='Similarity expects a text file')
            response = similarity(content_payload, msg)

        elif msg.get("task", {}).get("id") == "similarity_query":
            response = similarity_query(content_payload, source_payload, msg)
        else:
            return {"response": {}}
        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f'Processing failed: {str(e)}')



@app.get("/files/{filename:path}")
def serve_file(filename: str, background_tasks: BackgroundTasks):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail='File not found')

    def remove_file(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"Error deleting file {path}: {e}")

    #background_tasks.add_task(remove_file, file_path)
    #return FileResponse(file_path, background=background_tasks)
    return FileResponse(file_path)


def bow(content_text: str, msg: Dict) -> dict:
    texts = gensim.utils.simple_preprocess(content_text)
    print(texts)
    dictionary = gensim.corpora.Dictionary([texts])
    corpus = dictionary.doc2bow(texts)

    # Convert corpus to list of (word, count) tuples
    word_counts = []
    for word_id, count in corpus:
        word_counts.append({"word": dictionary[word_id], "count": count})


    
    # Sort by count in descending order
    sorted_counts = sorted(word_counts, key=lambda x: x["count"], reverse=True)
    print(sorted_counts)
    
    # Write sorted counts to JSON file
    output_uuid = str(uuid.uuid4())
    output_path = os.path.join(OUTPUT_FOLDER, f"{output_uuid}.bow.json")
    with open(output_path, 'w') as f:
        json.dump(sorted_counts, f, indent=2)
    return {"response": {"type": "stored", "uri": [f"/files/{output_uuid}.bow.json"]}}


def similarity(content_text: str, msg: Dict) -> dict:
    process_id = msg.get("process", {}).get("@rid", "").replace("#", "").replace(":", "_")
    outfile = f"{process_id}.tf-idf_matrix.tar"
    tar_path = Path(OUTPUT_FOLDER) / outfile
    index_path = Path(OUTPUT_FOLDER) / process_id

    # Reset builder state for a fresh index build
    builder.reset()
    builder.add_document(content_text)
    builder.build_similarity_index(directory=index_path, save_to_disk=True)
    builder.archive_directory_to_tar(source_dir=index_path, output_tar_path=tar_path)

    msg["response"] = {"file": {"type": "similarity_index", "label": outfile}}
    return {"response": {"type": "stored", "uri": [f"/files/{outfile}"]}, "message": msg}


def similarity_query(content_payload: bytes, source_text: str, msg: Dict) -> dict:
    import tarfile
    
    process_id = msg.get("process", {}).get("@rid", "").replace("#", "").replace(":", "_")
    print(f"Similarity query for process: {process_id}")
    
    extract_dir = Path(OUTPUT_FOLDER) / process_id
    

    # Save the tar file temporarily
    temp_tar = Path(OUTPUT_FOLDER) / f"{process_id}_temp.tar"
    with open(temp_tar, 'wb') as f:
        f.write(content_payload)
    
    # Extract the tar file
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(temp_tar, 'r') as tar:
        tar.extractall(extract_dir)
    
    os.remove(temp_tar)  # Clean up temp file
    
    # Load the similarity model from the extracted directory
    model_components = builder.load_similarity_model(str(extract_dir))
    print("Model loaded successfully")

    if not extract_dir.exists():
        raise HTTPException(status_code=404, detail='Similarity index not found. Please upload the index tar file first.')
    
    # Use the query_similarity method
    results = builder.query_similarity(msg, model_components, source_text)
    print(results)
    builder.clear_similarity_index(str(extract_dir))
    
    # Create output file with results
    outfile = f"{process_id}.similarity_results.json"
    output_path = Path(OUTPUT_FOLDER) / outfile
    print(output_path)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return {"response": {"type": "stored", "uri": [f"/files/{outfile}"]}}
    

    
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9009, limit_concurrency=2048) 
