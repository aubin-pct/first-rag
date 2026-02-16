import sys
import os

# Ajouter le dossier courant au sys.path pour permettre les imports relatifs (main, config, etc.)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from sqlalchemy.orm import Session
from rag_engine.service import setup_rag_system
from database import init_db, get_db
from models import ChatSession, ChatMessage
from langchain_core.messages import HumanMessage, AIMessage
from schemas import ChatRequest, ChatResponse, ChatSessionSchema, ChatMessageSchema
from typing import List
from fastapi import UploadFile, File
import shutil
from config import DATA_DIR
from rag_engine.loader import load_and_split_documents
from rag_engine.vector_store import index_documents

# Initialisation du système RAG (variable globale)
rag_system = None
retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    global rag_system, retriever
    print("🚀 Démarrage de l'API RAG...")
    try:
        rag_system, retriever = setup_rag_system()
        print("✅ Système RAG initialisé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation du RAG: {e}")
    
    yield
    
    # Shutdown
    print("🛑 Arrêt de l'API RAG...")

app = FastAPI(title="RAG API", description="API pour le système RAG", lifespan=lifespan)

# Configuration CORS pour autoriser les requêtes depuis l'application Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines (pour le dev)
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Autorise tous les headers
)

@app.get("/sessions", response_model=List[ChatSessionSchema])
def get_sessions(db: Session = Depends(get_db)):
    """Récupère la liste de toutes les sessions de chat, triées par épinglage puis date"""
    sessions = db.query(ChatSession).order_by(ChatSession.is_pinned.desc(), ChatSession.created_at.desc()).all()
    return sessions

@app.delete("/sessions/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db)):
    """Supprime une session de chat"""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    
    db.delete(session)
    db.commit()
    return {"message": "Session supprimée"}

@app.patch("/sessions/{session_id}/pin")
def toggle_pin_session(session_id: int, db: Session = Depends(get_db)):
    """Épingle ou désépingle une session"""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    
    session.is_pinned = not session.is_pinned
    db.commit()
    db.refresh(session)
    return {"message": "Statut mis à jour", "is_pinned": session.is_pinned}

@app.get("/sessions/{session_id}/messages", response_model=List[ChatMessageSchema])
def get_session_messages(session_id: int, db: Session = Depends(get_db)):
    """Récupère tous les messages d'une session spécifique"""
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp.asc()).all()
    if not messages:
        # On vérifie si la session existe quand même
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session non trouvée")
    return messages

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="Le système RAG n'est pas encore prêt")

    # 1. Gestion de la session de chat (DB)
    session = None
    if request.session_id:
        session = db.query(ChatSession).filter(ChatSession.id == request.session_id).first()
    
    if not session:
        # Nouvelle session
        session = ChatSession(title=request.question[:50] + "...")
        db.add(session)
        db.commit()
        db.refresh(session)

    # 2. Sauvegarde du message utilisateur
    user_msg = ChatMessage(
        session_id=session.id,
        role="user",
        content=request.question
    )
    db.add(user_msg)
    db.commit()

    try:
        # Conversion de l'historique JSON en objets LangChain
        chat_history = []
        for msg in request.history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        # Invocation du RAG
        response = rag_system.invoke({
            "input": request.question,
            "chat_history": chat_history
        })

        # Extraction du contexte (optionnel, pour debug ou affichage)
        context_docs = response.get("context", [])
        context_texts = [doc.page_content for doc in context_docs]

        print(f"\n🔍 Contexte récupéré ({len(context_docs)} chunks) :")
        for i, doc in enumerate(context_docs):
            print(f"--- Chunk {i+1} (Source: {doc.metadata.get('source', 'Inconnue')}) ---")
            print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            if 'relevance_score' in doc.metadata:
                print(f"Score: {doc.metadata['relevance_score']}")
            print("-" * 20)

        # 3. Sauvegarde de la réponse assistant
        assistant_msg = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=response["answer"]
        )
        db.add(assistant_msg)
        db.commit()

        return ChatResponse(
            answer=response["answer"],
            context=context_texts,
            session_id=session.id
        )
    except Exception as e:
        print(f"Erreur lors du chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global retriever
    if not retriever:
        raise HTTPException(status_code=503, detail="Le système RAG n'est pas encore prêt")
    
    # Sauvegarder le fichier dans DATA_DIR
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"📁 Fichier uploadé : {file.filename}")
    
    # Traiter le document
    from rag_engine.router import DocumentRouter
    router = DocumentRouter()
    documents = router.route_and_process(file_path)
    
    if documents:
        # Ajouter à la base vectorielle
        index_documents(retriever, documents)
        return {"message": f"Document '{file.filename}' ajouté avec succès. {len(documents)} fragments indexés."}
    else:
        return {"message": f"Échec du traitement du document '{file.filename}'."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
