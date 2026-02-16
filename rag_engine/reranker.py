from typing import Sequence, Any, List, Optional
from langchain_core.documents import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from pydantic import PrivateAttr
from FlagEmbedding import FlagReranker
from config import RERANKER_MODEL, MIN_RELEVANCE_SCORE

class BgeRerankCompressor(BaseDocumentCompressor):
    """
    Compresseur de documents utilisant FlagReranker (BGE-Reranker).
    Réordonne les documents en fonction de leur pertinence sémantique avec la requête.
    
    Filtre automatiquement les documents dont le score est inférieur au seuil MIN_RELEVANCE_SCORE.
    """
    model_name: str = RERANKER_MODEL
    top_n: int = 3
    min_score: Optional[float] = MIN_RELEVANCE_SCORE
    _reranker: Any = PrivateAttr()

    def __init__(self, model_name: str = RERANKER_MODEL, top_n: int = 3, 
                 min_score: Optional[float] = MIN_RELEVANCE_SCORE, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.top_n = top_n
        self.min_score = min_score
        print(f"🚀 Initialisation du Reranker : {model_name} (Cela peut prendre un moment...)")
        print(f"   ↳ Seuil de pertinence minimum : {min_score}")
        
        try:
            # use_fp16=True pour accélérer l'inférence sur GPU/CPU compatible
            self._reranker = FlagReranker(model_name, use_fp16=True)
        except Exception as e:
            print(f"⚠️ ERREUR : Impossible de charger le Reranker (BGE) : {e}")
            print("   ↳ Le système continuera de fonctionner sans reranking (recherche vectorielle/hybride seule).")
            self._reranker = None

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks=None
    ) -> Sequence[Document]:
        """
        Rerank les documents en utilisant le modèle BGE.
        """
        if not documents:
            return []
        
        # Si le reranker n'a pas pu être chargé (ex: hors ligne), on retourne les documents bruts
        if self._reranker is None:
            return documents
        
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Calculer les scores de pertinence
        scores = self._reranker.compute_score(pairs)
        
        # Gérer le cas où un seul document est passé (scores est un float)
        if isinstance(scores, float):
            scores = [scores]

        # Associer chaque document à son score
        doc_score_pairs = list(zip(documents, scores))
        
        # Trier par score décroissant (le plus pertinent en premier)
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        final_docs = []
        filtered_count = 0
        
        for doc, score in doc_score_pairs:
            if self.min_score is not None and score < self.min_score:
                filtered_count += 1
                continue
                
            doc.metadata["relevance_score"] = round(score, 3)
            final_docs.append(doc)
            
        return final_docs
