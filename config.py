import os
from dotenv import load_dotenv

load_dotenv()


LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

SEMANTIC_CHUNKER_THRESHOLD = 90  # Percentile : Top 10% des changements de sens = nouvelle section

CHUNK_SIZE = 4000 
CHUNK_OVERLAP = 200

SEARCH_K = 10 
USE_RERANKER = True
USE_HYBRID_SEARCH = True

# Seuil de pertinence du Reranker BGE:
# - Score > 0  : Très pertinent (match direct avec la requête)
# - Score -5 à 0 : Moyennement pertinent  
# - Score < -5 : Non pertinent (à filtrer)
MIN_RELEVANCE_SCORE = 0  # Ne garde que les documents vraiment pertinents 

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR)) 

DATA_DIR = os.path.join(BASE_DIR, "data") 
PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db") 
DOC_STORE_DIR = os.path.join(PROJECT_ROOT, "doc_store") 
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache") 
LLM_CACHE_DB = os.path.join(CACHE_DIR, "llm_cache.db")
EMBEDDINGS_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings_cache")
