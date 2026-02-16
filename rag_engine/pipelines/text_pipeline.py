import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import BasePipeline
from config import CHUNK_SIZE, CHUNK_OVERLAP

class TextPipeline(BasePipeline):
    """
    Pipeline B : 'The Fast Lane' (Documents Textuels)
    Utilise des loaders standards et un chunking sémantique/récursif.
    
    Chunking: Processus de découpage d'un long texte en segments plus courts ("chunks") pour faciliter leur traitement et leur indexation par le modèle.
    """
    
    def process(self, file_path: str) -> List[Document]:
        import time
        start_time = time.time()
        print(f"🏎️  Pipeline Texte activé pour : {os.path.basename(file_path)}")
        documents = []
        
        try:
            if file_path.lower().endswith(".txt"):
                loader = TextLoader(file_path)
                documents = loader.load()
            elif file_path.lower().endswith(".pdf"):
                print("   ↳ Chargement du PDF avec PyPDFLoader...")
                load_start = time.time()
                # Utiliser pypdf directement pour plus de contrôle
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                print(f"   ↳ PDF ouvert, {len(reader.pages)} pages détectées.")
                
                # Charger les pages par lots pour éviter la surcharge mémoire
                batch_size = 100
                all_text = ""
                for i in range(0, len(reader.pages), batch_size):
                    batch_end = min(i + batch_size, len(reader.pages))
                    print(f"   ↳ Traitement des pages {i+1} à {batch_end}...")
                    batch_start = time.time()
                    for page_num in range(i, batch_end):
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        if text.strip():
                            all_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
                    batch_end_time = time.time()
                    print(f"   ↳ Lot traité en {batch_end_time - batch_start:.2f}s")
                
                # Créer un document unique avec tout le texte
                from langchain_core.documents import Document
                documents = [Document(page_content=all_text, metadata={"source": file_path})]
                load_end = time.time()
                print(f"   ↳ PDF chargé en {load_end - load_start:.2f}s, 1 document créé.")
            
            # Chunking
            print("   ↳ Découpage en chunks...")
            chunk_start = time.time()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""] 
            )
            
            splits = text_splitter.split_documents(documents)
            chunk_end = time.time()
            print(f"   ↳ Découpage terminé en {chunk_end - chunk_start:.2f}s, {len(splits)} fragments générés.")
            
            total_time = time.time() - start_time
            print(f"   ↳ Pipeline Texte terminé en {total_time:.2f}s total.")
            return splits
            
        except Exception as e:
            print(f"❌ Erreur dans le pipeline texte : {e}")
            return []
