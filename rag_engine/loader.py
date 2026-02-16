import os
from typing import List
from langchain_core.documents import Document
from config import DATA_DIR
from .router import DocumentRouter

def load_and_split_documents() -> List[Document]:
    """
    Charge les documents en utilisant le Router Intelligent.
    
    Router: Composant logiciel qui analyse les métadonnées ou le contenu d'un fichier pour diriger son traitement vers le pipeline le plus approprié (ex: texte simple vs OCR pour images).
    """
    documents = []
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Dossier de données créé : {DATA_DIR} (Placez vos fichiers ici)")
        return []

    print(f"📂 Scan du dossier : {DATA_DIR}")
    
    router = DocumentRouter()
    
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        
        if filename.startswith("."):
            continue
            
        if os.path.isfile(file_path):
            docs = router.route_and_process(file_path)
            documents.extend(docs)
            
    return documents
