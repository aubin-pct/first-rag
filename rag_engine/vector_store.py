from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever, EnsembleRetriever
from langchain_ollama import OllamaEmbeddings
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import TextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List, Any
from langchain.embeddings import CacheBackedEmbeddings
from config import EMBEDDING_MODEL, PERSIST_DIR, DOC_STORE_DIR, SEARCH_K, USE_RERANKER, USE_HYBRID_SEARCH, EMBEDDINGS_CACHE_DIR, SEMANTIC_CHUNKER_THRESHOLD, MIN_RELEVANCE_SCORE
from .reranker import BgeRerankCompressor
import os
import shutil
import pickle


class SemanticTextSplitter(TextSplitter):
    """
    Wrapper qui adapte SemanticChunker à l'interface TextSplitter.
    Permet d'utiliser le découpage sémantique avec ParentDocumentRetriever.
    
    SemanticChunker: Découpe le texte aux endroits où le sens change significativement,
    en utilisant les embeddings pour détecter les transitions sémantiques.
    """
    
    def __init__(self, embeddings, breakpoint_threshold_type: str = "percentile", 
                 breakpoint_threshold_amount: int = 90, **kwargs):
        super().__init__(**kwargs)
        self._semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount
        )
    
    def split_text(self, text: str) -> List[str]:
        """Découpe le texte en utilisant SemanticChunker."""
        # SemanticChunker travaille avec des Documents, on crée un doc temporaire
        docs = self._semantic_chunker.create_documents([text])
        return [doc.page_content for doc in docs]

class ChildRerankingRetriever(BaseRetriever):
    """
    Retriever personnalisé qui récupère les chunks enfants, les reranke, puis remonte aux parents.
    
    Retriever: Composant chargé de retrouver les documents les plus pertinents dans une base de données en réponse à une requête.
    Reranker: Modèle de Deep Learning spécialisé qui réévalue et réordonne une liste de documents candidats pour améliorer la précision des résultats.
    """
    parent_retriever: ParentDocumentRetriever
    compressor: Any # BgeRerankCompressor
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # 1. Récupérer les enfants (candidats)
        search_kwargs = self.parent_retriever.search_kwargs
        k = search_kwargs.get("k", 4)
        
        children = self.parent_retriever.vectorstore.similarity_search(query, k=k)
        
        # 2. Reranking des enfants (avec filtrage par score)
        if not children:
            return []
            
        reranked_children = self.compressor.compress_documents(children, query)
        
        # 3. Récupération des parents uniques avec le meilleur score enfant
        parent_ids = []
        parent_best_scores = {}  # Garde le meilleur score pour chaque parent
        seen_ids = set()
        id_key = self.parent_retriever.id_key
        
        for child in reranked_children:
            doc_id = child.metadata.get(id_key)
            child_score = child.metadata.get("relevance_score", 0)
            
            if doc_id:
                # Mettre à jour le meilleur score pour ce parent
                if doc_id not in parent_best_scores or child_score > parent_best_scores[doc_id]:
                    parent_best_scores[doc_id] = child_score
                    
                if doc_id not in seen_ids:
                    parent_ids.append(doc_id)
                    seen_ids.add(doc_id)
        
        # 4. Fetch des parents et ajout du score de pertinence
        if not parent_ids:
            return []
            
        parents = self.parent_retriever.docstore.mget(parent_ids)
        final_parents = []
        for parent, doc_id in zip(parents, parent_ids):
            if parent is not None:
                # Propager le meilleur score enfant au parent
                parent.metadata["relevance_score"] = parent_best_scores.get(doc_id, 0)
                final_parents.append(parent)
                
        return final_parents

def get_vectorstore():
    """
    Récupère ou initialise la base vectorielle Chroma avec Cache d'Embeddings.
    
    Vector Store: Base de données optimisée pour stocker et rechercher des vecteurs (représentations mathématiques du texte).
    Embedding: Processus de conversion d'un texte en un vecteur numérique de dimension fixe, capturant son sens sémantique.
    """
    # 1. Modèle d'embedding de base
    base_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # 2. Configuration du cache pour les embeddings
    if not os.path.exists(EMBEDDINGS_CACHE_DIR):
        os.makedirs(EMBEDDINGS_CACHE_DIR)
        
    store = LocalFileStore(EMBEDDINGS_CACHE_DIR)
    
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        base_embeddings, 
        store, 
        namespace=EMBEDDING_MODEL
    )
    
    print(f"⚡ Cache d'embeddings activé : {EMBEDDINGS_CACHE_DIR}")

    return Chroma(
        collection_name="full_documents",
        persist_directory=PERSIST_DIR, 
        embedding_function=cached_embeddings
    )

def get_docstore():
    """
    Récupère ou initialise le stockage des documents parents.
    
    DocStore: Système de stockage persistant (clé-valeur) conservant les documents originaux complets, par opposition aux vecteurs.
    """
    if not os.path.exists(DOC_STORE_DIR):
        os.makedirs(DOC_STORE_DIR)
    
    # Stockage physique (bytes) sur le disque
    fs = LocalFileStore(DOC_STORE_DIR)
    
    # Wrapper pour gérer la sérialisation des Documents (Document -> bytes)
    return EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x, # Les clés (UUIDs) sont déjà des strings sûres
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads
    )

def get_all_documents_from_store(docstore):
    """Récupère tous les documents du docstore pour initialiser BM25"""
    docs = []
    keys = list(docstore.yield_keys())
    if not keys:
        return []
    
    results = docstore.mget(keys)
    for res in results:
        if res:
            docs.append(res)
    return docs

def get_retriever(vectorstore, docstore):
    """
    Retourne le retriever final avec architecture optimisée:
    
    1. BM25 (mots-clés) + Vectoriel (sémantique) → EnsembleRetriever
    2. Reranker BGE appliqué SUR LE RÉSULTAT FINAL pour filtrer et réordonner
    
    Cette architecture garantit que:
    - Les recherches par mot-clé (ex: "eytan") sont bien prises en compte par BM25
    - Les recherches sémantiques sont gérées par le vectoriel
    - Le reranker filtre les résultats non pertinents APRÈS la fusion
    """
    # 1. Configuration du ParentDocumentRetriever (Base Vectorielle)
    base_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    child_splitter = SemanticTextSplitter(
        embeddings=base_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=SEMANTIC_CHUNKER_THRESHOLD
    )

    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        search_kwargs={"k": SEARCH_K * 3}  # On récupère plus de candidats pour le reranking final
    )

    # 2. Construction du retriever de base (vectoriel ou hybride)
    base_retriever = parent_retriever
    
    if USE_HYBRID_SEARCH:
        existing_docs = get_all_documents_from_store(docstore)
        
        if existing_docs:
            print(f"🔀 Activation de la Recherche Hybride (BM25 + Vector) - {len(existing_docs)} documents")
            bm25_retriever = BM25Retriever.from_documents(existing_docs)
            bm25_retriever.k = SEARCH_K * 2  # Plus de candidats BM25
            
            # Ensemble: BM25 (40%) + Vectoriel (60%)
            base_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, parent_retriever],
                weights=[0.4, 0.6]  # BM25 renforcé pour les recherches par mot-clé
            )
        else:
            print("⚠️ Pas de documents pour BM25, retour au vectoriel seul.")

    # 3. Application du Reranker SUR LE RÉSULTAT FINAL (filtrage + réordonnancement)
    if USE_RERANKER:
        print(f"✨ Activation du Reranker BGE FINAL (Top {SEARCH_K}, seuil: {MIN_RELEVANCE_SCORE})")
        compressor = BgeRerankCompressor(top_n=SEARCH_K)
        
        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        return final_retriever
            
    return base_retriever


def _extract_parent_retriever(retriever):
    """
    Extrait le ParentDocumentRetriever depuis n'importe quelle structure de retriever.
    Gère les cas: ContextualCompressionRetriever -> EnsembleRetriever -> ParentDocumentRetriever
    """
    # Cas 1: ContextualCompressionRetriever (reranker final)
    if isinstance(retriever, ContextualCompressionRetriever):
        return _extract_parent_retriever(retriever.base_retriever)
    
    # Cas 2: EnsembleRetriever (BM25 + Vector)
    if isinstance(retriever, EnsembleRetriever):
        for r in retriever.retrievers:
            if isinstance(r, ParentDocumentRetriever):
                return r
            # Récursion si nécessaire
            result = _extract_parent_retriever(r)
            if result:
                return result
        return None
    
    # Cas 3: ParentDocumentRetriever directement
    if isinstance(retriever, ParentDocumentRetriever):
        return retriever
    
    return None


def index_documents(retriever, documents):
    """
    Indexe les documents dans le ParentDocumentRetriever.
    Extrait automatiquement le bon retriever depuis n'importe quelle structure.
    """
    import time
    start_time = time.time()
    print(f"🏗️  Indexation de {len(documents)} documents parents...")
    
    parent_retriever = _extract_parent_retriever(retriever)
    
    if not parent_retriever:
        raise ValueError("Impossible de trouver le ParentDocumentRetriever sous-jacent")

    # Indexation par lots pour éviter la surcharge
    batch_size = 100
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = i // batch_size + 1
        batch_start = time.time()
        
        print(f"   ↳ Indexation du lot {batch_num}/{total_batches} ({len(batch)} documents)...")
        parent_retriever.add_documents(batch, ids=None)
        
        batch_end = time.time()
        elapsed = batch_end - start_time
        eta = (elapsed / batch_num) * (total_batches - batch_num)
        print(f"   ↳ Lot {batch_num} terminé en {batch_end - batch_start:.2f}s. Temps écoulé: {elapsed:.2f}s, ETA: {eta:.2f}s")

    total_time = time.time() - start_time
    print(f"✅ Indexation terminée en {total_time:.2f}s.")
