import gensim
from gensim import corpora, models, similarities
from time import time
import logging
import pickle # <-- Required to save/load non-Gensim Python objects
import tarfile
import os
from pathlib import Path
import shutil

# --- CONFIGURABLE PARAMETERS FOR SLIDING WINDOW ---
WINDOW_SIZE = 15  # Number of tokens in each document chunk
OVERLAP = 5       # Number of tokens that consecutive chunks overlap

# --- FILE PATHS FOR PERSISTENCE ---
# Gensim components must be saved to disk to be reused later.
DICT_FILE = "dictionary.dict"
TFIDF_FILE = "tfidf.model"
INDEX_FILE = "matrix_similarity.index"
# Python objects required for mapping
CHUNK_MAP_FILE = "original_chunk_map.pkl"
CHUNK_START_MAP_FILE = "original_token_start_map.pkl" # New map for location tracking
CORPUS_CHUNKS_FILE = "corpus_chunks_tokenized.pkl"  # Tokenized chunks for alignment

# Set up logging for Gensim messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def find_char_start_index(original_text, start_token_index):
    """
    Calculates the character index in the original text corresponding to a token start index.
    
    This is necessary because simple_tokenize() is a destructive operation (removes 
    punctuation, lowercases). This function re-tokenizes the text while tracking 
    character positions to accurately map token indices back to character positions.
    
    Args:
        original_text (str): The original untokenized text
        start_token_index (int): The token index to find the character position for
        
    Returns:
        int: Character index where the token starts, or -1 if not found
    """
    
    if not original_text:
        return -1
    
    if start_token_index < 0:
        return -1
    
    # Tokenize to get expected tokens (this is what gensim does)
    expected_tokens = simple_tokenize(original_text)
    
    if start_token_index >= len(expected_tokens):
        return -1
    
    # Simulate tokenization character by character to track positions
    token_starts = []  # Character start position for each token
    current_word = ""
    current_word_start = -1
    token_idx = 0
    
    i = 0
    while i < len(original_text) and token_idx <= start_token_index:
        char = original_text[i]
        
        # If it's an alphanumeric character, it's part of a word
        if char.isalnum():
            if current_word_start == -1:
                current_word_start = i  # Mark where this word started
            current_word += char.lower()  # Build word in lowercase
        else:
            # Non-alphanumeric - we've reached the end of a potential word
            if current_word and len(current_word) >= 2:  # simple_preprocess filters < 2 chars
                # Check if this matches the next expected token
                if token_idx < len(expected_tokens) and current_word == expected_tokens[token_idx]:
                    token_starts.append(current_word_start)
                    # If this is our target token, we can return early
                    if token_idx == start_token_index:
                        return current_word_start
                    token_idx += 1
            # Reset for next word
            current_word = ""
            current_word_start = -1
        
        i += 1
    
    # Handle the last word if text doesn't end with non-alphanumeric
    if current_word and len(current_word) >= 2:
        if token_idx < len(expected_tokens) and current_word == expected_tokens[token_idx]:
            token_starts.append(current_word_start)
            if token_idx == start_token_index:
                return current_word_start
            token_idx += 1
    
    # If we successfully tracked tokens up to our target
    if start_token_index < len(token_starts):
        return token_starts[start_token_index]
    
    return -1

def simple_tokenize(text):
    """Tokenize text to prepare for chunking."""
    return gensim.utils.simple_preprocess(text)

def find_token_alignment(query_chunk, corpus_chunk):
    """
    Find where query chunk tokens align with corpus chunk tokens.
    
    Returns the offset (in tokens) where query starts within corpus chunk,
    or 0 if no good alignment is found (defaults to start of chunk).
    """
    if not query_chunk or not corpus_chunk:
        return 0
    
    # Try to find where query tokens start in corpus chunk
    # Look for the first query token in the corpus chunk
    first_query_token = query_chunk[0]
    
    # Find all occurrences of first query token in corpus chunk
    for i, token in enumerate(corpus_chunk):
        if token == first_query_token:
            # Verify that subsequent tokens also match
            matches = 0
            for j in range(min(len(query_chunk), len(corpus_chunk) - i)):
                if j < len(query_chunk) and i + j < len(corpus_chunk):
                    if query_chunk[j] == corpus_chunk[i + j]:
                        matches += 1
                    else:
                        break
            
            # If we have a good match (at least first 3 tokens or 50% of query), use this offset
            if matches >= min(3, len(query_chunk) * 0.5):
                return i
    
    # If no good alignment found, default to 0 (start of chunk)
    return 0

def sliding_window_tokenize(text, window_size, overlap):
    """
    Splits a document into a list of tokenized, overlapping chunks.
    
    This function is primarily used here for the query document, as the 
    corpus chunking is handled explicitly in Step 2 to record the start indices.
    
    Args:
        text (str): The input document text.
        window_size (int): The number of tokens in each chunk.
        overlap (int): The number of tokens to overlap between chunks.
        
    Returns:
        list: A list of tokenized chunks (list of lists of strings).
    """
    tokens = simple_tokenize(text)
    step = window_size - overlap
    
    # Ensure step is positive and not too large
    if step <= 0:
        step = 1 # Fallback to minimal movement
        
    chunks = []
    # Slide the window across the tokens
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + window_size]
        if chunk:
            chunks.append(chunk)
            
    return chunks



class IncrementalSimilarityBuilder:
    """
    Builder class for incrementally adding documents to a similarity index.
    Tracks chunks, maps, and allows building the final similarity index when ready.
    """
    
    def __init__(self, window_size=15, overlap=5):
        """
        Initialize the builder with chunking parameters.
        
        Args:
            window_size (int): Number of tokens in each document chunk.
            overlap (int): Number of tokens that consecutive chunks overlap.
        """
        self.window_size = window_size
        self.overlap = overlap
        self.reset()
    
    def reset(self):
        """Reset the builder state for a new index build."""
        self.documents = []
        self.corpus_chunks_tokenized = []
        self.original_chunk_map = []
        self.original_token_start_map = []
        self.doc_count = 0
        
    def add_document(self, text):
        """
        Add a single document to the corpus and update all tracking maps.
        
        Args:
            text (str): The document text to add.
        """
        tokens = simple_tokenize(text)
        step = self.window_size - self.overlap
        
        if step <= 0:
            step = 1
        
        # Chunk the document and update all tracking structures
        for i in range(0, len(tokens), step):
            chunk = tokens[i:i + self.window_size]
            if chunk:
                self.corpus_chunks_tokenized.append(chunk)
                self.original_chunk_map.append(self.doc_count)
                self.original_token_start_map.append(i)
        
        self.documents.append(text)
        self.doc_count += 1
        print(f"Added document {self.doc_count} with {len(tokens)} tokens")
        
    def build_similarity_index(self, directory, save_to_disk=True):
        """
        Build the final similarity index from all accumulated documents.
        
        Args:
            save_to_disk (bool): Whether to save all models to disk.
            
        Returns:
            dict: Dictionary containing all built components:
                - dictionary: Gensim Dictionary
                - tfidf_model: TF-IDF Model
                - index: MatrixSimilarity Index
                - corpus_chunks_tokenized: All tokenized chunks
                - original_chunk_map: Map from chunk to document
                - original_token_start_map: Map from chunk to starting token index
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not self.corpus_chunks_tokenized:
            raise ValueError("No documents have been added to the index")
        
        print(f"\n--- Building similarity index from {self.doc_count} documents ---")
        print(f"Total chunks: {len(self.corpus_chunks_tokenized)}")
        
        # Create dictionary from all chunks
        print("Creating dictionary...")
        dictionary = corpora.Dictionary(self.corpus_chunks_tokenized)
        
        # Save dictionary
        if save_to_disk:
            dictionary.save(os.path.join(directory, DICT_FILE))
            print(f"Dictionary saved to {DICT_FILE}")
        
        # Convert chunks to Bag-of-Words
        print("Converting chunks to Bag-of-Words...")
        corpus_bow = [dictionary.doc2bow(chunk) for chunk in self.corpus_chunks_tokenized]
        
        # Train TF-IDF model
        print("Training TF-IDF model...")
        start_time = time()
        tfidf_model = models.TfidfModel(corpus_bow)
        
        # Save TF-IDF model
        if save_to_disk:
            tfidf_model.save(os.path.join(directory, TFIDF_FILE))
            print(f"TF-IDF Model saved to {TFIDF_FILE}")
        
        # Apply TF-IDF transformation
        print("Applying TF-IDF transformation...")
        corpus_tfidf = tfidf_model[corpus_bow]
        
        # Create similarity index
        print("Creating similarity index...")
        index = similarities.MatrixSimilarity(corpus_tfidf, num_features=len(dictionary))
        
        if save_to_disk:
            index.save(os.path.join(directory, INDEX_FILE))
            print(f"MatrixSimilarity Index saved to {INDEX_FILE}")
            
            # Save tracking maps
            with open(os.path.join(directory, CHUNK_MAP_FILE), 'wb') as f:
                pickle.dump(self.original_chunk_map, f)
            print(f"Original chunk map saved to {CHUNK_MAP_FILE}")
            
            with open(os.path.join(directory, CHUNK_START_MAP_FILE), 'wb') as f:
                pickle.dump(self.original_token_start_map, f)
            print(f"Original token start map saved to {CHUNK_START_MAP_FILE}")
            
            # Save corpus chunks for alignment calculation during queries
            with open(os.path.join(directory, CORPUS_CHUNKS_FILE), 'wb') as f:
                pickle.dump(self.corpus_chunks_tokenized, f)
            print(f"Corpus chunks tokenized saved to {CORPUS_CHUNKS_FILE}")
        
        end_time = time()
        print(f"Index built in: {end_time - start_time:.4f} seconds.")
        
        return {
            'dictionary': dictionary,
            'tfidf_model': tfidf_model,
            'index': index,
            'corpus_chunks_tokenized': self.corpus_chunks_tokenized,
            'original_chunk_map': self.original_chunk_map,
            'original_token_start_map': self.original_token_start_map,
            'documents': self.documents
        }


    def archive_directory_to_tar(self, source_dir, output_tar_path):
        """
        Archives all files from a given directory into an uncompressed tar file.
        
        Args:
            source_dir (str): Path to the directory containing files to archive.
            output_tar_path (str): Path to the output tar file.
            
        Returns:
            str: Path to the created tar file.
            
        Raises:
            FileNotFoundError: If source_dir doesn't exist.
            Exception: For other archiving errors.
        """
        print(f"Archiving directory {source_dir} to {output_tar_path}")
        source_path = Path(source_dir)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Directory '{source_dir}' does not exist.")
        
        if not source_path.is_dir():
            raise ValueError(f"'{source_dir}' is not a directory.")

        print(f"Source path: {source_path}")
        
        # Create the uncompressed tar file
        with tarfile.open(output_tar_path, 'w', format=tarfile.USTAR_FORMAT) as tar:
            # Walk through all files in the source directory
            for file_path in source_path.rglob('*'):
                if file_path.is_file():
                    # Add the file to the tar with just the filename (no directory path)
                    arcname = file_path.name
                    tar.add(str(file_path), arcname=arcname)
                    print(f"Added: {file_path} as {arcname}")
                    # delete the file after adding it to the tar
                    os.remove(file_path)
        
        print(f"Successfully created tar archive: {output_tar_path}")
        return output_tar_path



    def query_similarity(self, msg, model_components=None, source_text=None):
        """
        Perform a similarity query against a loaded model.
        
        Args:
            query_text (str): The text to query for similarity.
            model_components (dict, optional): Loaded model components. If None, will use 
                stored components from the builder.
            
        Returns:
            dict: Dictionary containing similarity results:
                - max_similarity: Maximum similarity score found
                - average_similarity: Average similarity across all chunks
                - chunk_count: Number of chunks in the query
                - chunk_similarities: List of similarity scores per chunk
                - best_match_doc_index: Document index with best match
                - best_match_token_start: Token start position of best match
        """
        query_text = msg.get("task", {}).get("params", {}).get("query", "")
        if not query_text:
            return {"error": "No query text provided"}

        if model_components is None:
            # Use stored components if available
            if not hasattr(self, '_loaded_components'):
                raise ValueError("No model components loaded. Call load_similarity_model() first.")
            model_components = self._loaded_components
        
        # Extract components
        dictionary = model_components['dictionary']
        tfidf_model = model_components['tfidf_model']
        index = model_components['index']
        original_chunk_map = model_components['original_chunk_map']
        original_token_start_map = model_components['original_token_start_map']
        # Get corpus chunks if available (for alignment calculation)
        corpus_chunks_tokenized = model_components.get('corpus_chunks_tokenized', None)
        
        # Validate source_text is provided
        if not source_text:
            return {"error": "source_text parameter is required for character position mapping"}
        
        print(f"\n--- Performing similarity query ---")
        
        # Tokenize and chunk the query
        query_chunks = sliding_window_tokenize(query_text, self.window_size, self.overlap)
        print(f"Query text tokenized into {len(query_chunks)} chunks")
        
        # Convert to bag-of-words and TF-IDF
        query_bow = [dictionary.doc2bow(chunk) for chunk in query_chunks]
        query_tfidf = [tfidf_model[bow] for bow in query_bow]
        
        # Find similarities for each chunk
        chunk_similarities = []
        max_similarity = 0
        best_match_doc_index = -1
        best_match_token_start = -1
        best_match_query_start_token = -1
        
        for q_chunk_idx, query_chunk_tfidf in enumerate(query_tfidf):
            chunk_sims = index[query_chunk_tfidf]
            chunk_sims_list = chunk_sims.tolist()

            # Find the best match for this query chunk (highest similarity score)
            threshold = 0.3
            max_chunk_score = max(chunk_sims_list) if chunk_sims_list else 0
            
            if max_chunk_score > threshold:
                # Find which corpus chunk achieved this max score
                best_corpus_chunk_index = chunk_sims_list.index(max_chunk_score)
                doc_index = original_chunk_map[best_corpus_chunk_index]
                chunk_start_token = original_token_start_map[best_corpus_chunk_index]
                query_chunk_tokens = query_chunks[q_chunk_idx]
                
                # Find where the query actually starts within the matched corpus chunk
                # This accounts for partial matches or alignment within overlapping chunks
                if corpus_chunks_tokenized and best_corpus_chunk_index < len(corpus_chunks_tokenized):
                    corpus_chunk_tokens = corpus_chunks_tokenized[best_corpus_chunk_index]
                    alignment_offset = find_token_alignment(query_chunk_tokens, corpus_chunk_tokens)
                else:
                    # If corpus chunks not available, default to 0 (start of chunk)
                    alignment_offset = 0
                
                # Calculate the actual token start position accounting for alignment
                token_start = chunk_start_token + alignment_offset
                
                # Use source_text directly (the original indexed document text)
                char_start = find_char_start_index(source_text, token_start)
                
                chunk_similarities.append({
                    "doc_index": doc_index,
                    "similarity": float(max_chunk_score),
                    "text_start_char": char_start,  #char start index in the original text
                    "text_start_token": token_start,
                    "query_start_token": q_chunk_idx * (self.window_size - self.overlap)
                })
            
            # Track the best match across all chunks (for best_match fields)
            if max_chunk_score > max_similarity and max_chunk_score > threshold:
                max_similarity = max_chunk_score
                print(max_similarity)
                # Find which corpus chunk achieved this max score
                best_corpus_chunk_index = chunk_sims_list.index(max_chunk_score)
                best_match_doc_index = original_chunk_map[best_corpus_chunk_index]
                chunk_start_token = original_token_start_map[best_corpus_chunk_index]
                query_chunk_tokens = query_chunks[q_chunk_idx]
                
                # Find alignment for best match too
                if corpus_chunks_tokenized and best_corpus_chunk_index < len(corpus_chunks_tokenized):
                    corpus_chunk_tokens = corpus_chunks_tokenized[best_corpus_chunk_index]
                    alignment_offset = find_token_alignment(query_chunk_tokens, corpus_chunk_tokens)
                else:
                    alignment_offset = 0
                
                best_match_token_start = chunk_start_token + alignment_offset
                best_match_query_start_token = q_chunk_idx * (self.window_size - self.overlap)  

        if best_match_doc_index != -1 and best_match_token_start != -1:
            # Use source_text directly (the original indexed document text)
            best_match_char_start = find_char_start_index(source_text, best_match_token_start)
   

        result = {
            "query_text": query_text,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "max_similarity": float(max_similarity),
            "chunk_count": len(query_chunks),
            "chunk_similarities": chunk_similarities,
            "best_match_doc_index": int(best_match_doc_index),
            "best_match_token_start": int(best_match_token_start),
            "best_match_query_start_token": int(best_match_query_start_token)
        }
        
        # Add character start position if calculated
        if best_match_doc_index != -1 and best_match_token_start != -1:
            result["best_match_char_start"] = int(best_match_char_start) if best_match_char_start != -1 else -1
        source_text_file = msg.get("file", {}).get("source", {}).get("@rid", "")
        result["doc_map"] = [source_text_file]
        
        return result



    def load_similarity_model(self, source_dir):
        """
        Load a previously saved similarity model from a directory.
        
        Args:
            source_dir (str): Directory containing the model files.
            
        Returns:
            dict: Dictionary containing all loaded components:
                - dictionary: Gensim Dictionary
                - tfidf_model: TF-IDF Model
                - index: MatrixSimilarity Index
                - original_chunk_map: Map from chunk to original document index
                - original_token_start_map: Map from chunk to starting token index
        
        Raises:
            FileNotFoundError: If source_dir doesn't exist or any required file is missing.
            Exception: For other loading errors.
        """
        source_path = Path(source_dir)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Directory '{source_dir}' does not exist.")
        
        if not source_path.is_dir():
            raise ValueError(f"'{source_dir}' is not a directory.")
        
        # Build full paths to all files
        dict_file = source_path / DICT_FILE
        tfidf_file = source_path / TFIDF_FILE
        index_file = source_path / INDEX_FILE
        chunk_map_file = source_path / CHUNK_MAP_FILE
        chunk_start_map_file = source_path / CHUNK_START_MAP_FILE
        corpus_chunks_file = source_path / CORPUS_CHUNKS_FILE
        
        print(f"\n--- Loading Similarity Model Components from {source_dir} ---")
        
        # Load Gensim components
        try:
            print(f"Loading dictionary from {dict_file}...")
            dictionary = corpora.Dictionary.load(str(dict_file))
            print(f"Loading TF-IDF model from {tfidf_file}...")
            tfidf_model = models.TfidfModel.load(str(tfidf_file))
            print(f"Loading similarity index from {index_file}...")
            index = similarities.MatrixSimilarity.load(str(index_file))
        except Exception as e:
            raise FileNotFoundError(f"Failed to load Gensim components: {e}")
        
        # Load Python objects (maps)
        try:
            print(f"Loading chunk map from {chunk_map_file}...")
            with open(chunk_map_file, 'rb') as f:
                original_chunk_map = pickle.load(f)
            
            print(f"Loading token start map from {chunk_start_map_file}...")
            with open(chunk_start_map_file, 'rb') as f:
                original_token_start_map = pickle.load(f)
            
            # Load corpus chunks if available (for alignment)
            corpus_chunks_tokenized = None
            if corpus_chunks_file.exists():
                print(f"Loading corpus chunks from {corpus_chunks_file}...")
                with open(corpus_chunks_file, 'rb') as f:
                    corpus_chunks_tokenized = pickle.load(f)
            else:
                print(f"Warning: {corpus_chunks_file} not found, alignment calculation will be limited")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load mapping files: {e}")
        
        print("All components loaded successfully!")
        
        components = {
            'dictionary': dictionary,
            'tfidf_model': tfidf_model,
            'index': index,
            'original_chunk_map': original_chunk_map,
            'original_token_start_map': original_token_start_map,
            'corpus_chunks_tokenized': corpus_chunks_tokenized
        }
        
        # Store components for later use in query_similarity
        self._loaded_components = components
        
        return components

    def clear_similarity_index(self, source_dir):
        """
        Clear the similarity index from a directory.
        """
        source_path = Path(source_dir)
        if source_path.exists():
            shutil.rmtree(source_path)


# Create a builder instance
# builder = IncrementalSimilarityBuilder(window_size=15, overlap=5)

# # Add documents one by one
# builder.add_document("First document with some text content.")
# builder.add_document("Second document with different content.")
# builder.add_document("Third document with more text to search through.")

# result = builder.build_similarity_index(directory="output/similarity", save_to_disk=True)

# --- EXAMPLE: Using IncrementalSimilarityBuilder for one-by-one document addition ---
"""
Example usage of the incremental builder:

# Create a builder instance
builder = IncrementalSimilarityBuilder(window_size=15, overlap=5)

# Add documents one by one
builder.add_document("First document with some text content.")
builder.add_document("Second document with different content.")
builder.add_document("Third document with more text to search through.")

# Build the final similarity index (and save to disk)
result = builder.build_similarity_index(directory="similarity", save_to_disk=True)

# Now you can use the index for similarity searches
dictionary = result['dictionary']
tfidf_model = result['tfidf_model']
index = result['index']

# Query a document for similarity
query_text = "text to search for similarities"
query_chunks = sliding_window_tokenize(query_text, WINDOW_SIZE, OVERLAP)
query_bow = [dictionary.doc2bow(chunk) for chunk in query_chunks]
query_tfidf = [tfidf_model[bow] for bow in query_bow]

# Find similarities
for query_chunk_tfidf in query_tfidf:
    similarities = index[query_chunk_tfidf]
    max_sim = max(similarities)
    print(f"Max similarity: {max_sim:.4f}")
"""
