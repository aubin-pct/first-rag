import os
from typing import List
from langchain_core.documents import Document
from .base import BasePipeline
from config import LLAMA_CLOUD_API_KEY

# Import conditionnel
try:
    from llama_parse import LlamaParse
    HAS_LLAMA_PARSE = True
except ImportError:
    HAS_LLAMA_PARSE = False

class VisionPipeline(BasePipeline):
    """
    Pipeline A : 'The Sniper' (Documents Visuels/Techniques)
    Utilise LlamaParse (mode premium/vision) ou une approche OCR avancée.
    Pour l'instant, on utilise LlamaParse comme proxy pour la 'Vision' car il gère très bien les tables.
    """
    
    def process(self, file_path: str) -> List[Document]:
        """
        Traite le document avec LlamaParse pour extraire la structure complexe.
        """
        print(f"🦅 Pipeline Vision (Sniper) activé pour : {os.path.basename(file_path)}")
        
        if not HAS_LLAMA_PARSE or not LLAMA_CLOUD_API_KEY:
            print("⚠️ LlamaParse non disponible ou clé manquante. Fallback sur TextPipeline.")
            from .text_pipeline import TextPipeline
            return TextPipeline().process(file_path)

        try:
            # Configuration de LlamaParse pour extraire le markdown avec analyse de layout
            parser = LlamaParse(
                api_key=LLAMA_CLOUD_API_KEY,
                result_type="markdown",
                verbose=True,
                system_prompt="""
                    You are parsing a document for a RAG system.
                    CRITICAL INSTRUCTION: completely remove and ignore the Table of Contents (Sommaire), 
                    Index, and List of Figures. Do not output them in the result. 
                    Focus only on the content chapters.
                """
            )
            
            print("   ↳ Envoi à LlamaCloud pour analyse structurelle...")
            documents = parser.load_data(file_path)
            
            if not documents:
                raise ValueError("Aucun document extrait par LlamaParse (possiblement erreur de crédits ou fichier incompatible)")
            
            from langchain_text_splitters import MarkdownHeaderTextSplitter
            
            # Découpage par headers Markdown (Structurel)
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            
            final_docs = []
            for doc in documents:
                # Si c'est un objet LlamaIndex, il a un attribut 'text' ou 'get_content()'
                content = getattr(doc, "text", str(doc))
                
                # Split markdown
                md_splits = markdown_splitter.split_text(content)
                for split in md_splits:
                    # Construction du chemin hiérarchique (ex: "Contrat > Article 4 > Résiliation")
                    header_path_parts = []
                    for header_key in ["Header 1", "Header 2", "Header 3"]:
                        if header_key in split.metadata:
                            header_path_parts.append(split.metadata[header_key])
                    
                    # CRUCIAL : Préfixer le contenu avec le chemin du titre pour garder le contexte
                    if header_path_parts:
                        header_path = " > ".join(header_path_parts)
                        split.page_content = f"[{header_path}]\n\n{split.page_content}"
                        split.metadata["header_path"] = header_path
                    
                    # On rajoute la source
                    split.metadata["source"] = file_path
                    split.metadata["pipeline"] = "vision"
                    final_docs.append(split)
            
            print(f"   ↳ {len(final_docs)} fragments structurels (PARENTS avec chemin hiérarchique) générés.")
            return final_docs

        except Exception as e:
            print(f"❌ Erreur dans le pipeline vision : {e}")
            print("   ↳ Fallback sur TextPipeline pour extraction basique...")
            from .text_pipeline import TextPipeline
            return TextPipeline().process(file_path)
