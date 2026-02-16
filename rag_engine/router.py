import os
from typing import List
from langchain_core.documents import Document
from .pipelines.text_pipeline import TextPipeline
from .pipelines.vision_pipeline import VisionPipeline

class DocumentRouter:
    """
    Le Gardien (Router Agent).
    Décide quel pipeline utiliser pour chaque document.
    
    Pipeline: Chaîne de traitement séquentielle où la sortie d'une étape devient l'entrée de la suivante (ex: Chargement -> Nettoyage -> Découpage).
    """
    
    def __init__(self):
        self.text_pipeline = TextPipeline()
        self.vision_pipeline = VisionPipeline()

    def route_and_process(self, file_path: str) -> List[Document]:
        """
        Analyse le document et choisit le bon pipeline.
        """
        filename = os.path.basename(file_path)
        
        # Heuristique simple pour commencer (Extension)
        if file_path.lower().endswith(".txt"):
            return self.text_pipeline.process(file_path)
        
        # Pour le moment, on force le VisionPipeline pour les PDF car c'est le but de l'upgrade
        if file_path.lower().endswith(".pdf"):
            print(f"🚦 Router: PDF détecté '{filename}' -> Direction Pipeline Vision")
            return self.vision_pipeline.process(file_path)
            
        return self.text_pipeline.process(file_path)

    def _analyze_complexity(self, file_path: str) -> bool:
        """
        (À implémenter) Convertit la page 1 en image et demande à un LLM/Classifier.
        """
        return True
