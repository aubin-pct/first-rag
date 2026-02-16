from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BasePipeline(ABC):
    """
    Classe abstraite pour les pipelines d'ingestion.
    """
    
    @abstractmethod
    def process(self, file_path: str) -> List[Document]:
        """
        Traite un fichier et retourne une liste de documents (chunks).
        """
        pass
