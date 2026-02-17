from .loader import load_and_split_documents
from .vector_store import get_vectorstore, get_docstore, get_retriever, index_documents
from .chain import create_rag_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
import os
from config import PERSIST_DIR, DOC_STORE_DIR, LLM_CACHE_DB, CACHE_DIR

def setup_rag_system():
    """
    Configure et retourne le système RAG.
    
    RAG (Retrieval-Augmented Generation): Technique d'IA qui améliore les réponses d'un LLM en lui fournissant des informations pertinentes récupérées dans une base de connaissances externe avant de générer sa réponse.
    Cache: Mécanisme de stockage temporaire permettant de sauvegarder les résultats de calculs coûteux (comme les réponses du LLM) pour les réutiliser rapidement lors de requêtes identiques.
    """
    print("🔧 Configuration du système RAG avec Groq...")

    # 0. Initialisation du Cache LLM
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    set_llm_cache(SQLiteCache(database_path=LLM_CACHE_DB))
    print(f"🧠 Cache LLM activé : {LLM_CACHE_DB}")

    # 1. Initialisation des composants de stockage
    vectorstore = get_vectorstore()
    docstore = get_docstore()
    
    # 2. Vérifier si la base est vide (avant de créer le retriever)
    is_empty = not os.path.exists(DOC_STORE_DIR) or not os.listdir(DOC_STORE_DIR)
    
    # 3. Création du Retriever (Parent-Child) - première passe
    retriever = get_retriever(vectorstore, docstore)

    # 4. Indexation si nécessaire
    if is_empty:
        print("📂 Base de documents vide. Lancement de l'ingestion...")
        documents = load_and_split_documents()
        if documents:
            index_documents(retriever, documents)
            # Recréer le retriever APRÈS indexation pour que BM25 ait accès aux docs
            print("🔄 Reconfiguration du retriever avec les nouveaux documents...")
            retriever = get_retriever(vectorstore, docstore)
        else:
            print("⚠️ Aucun document trouvé à indexer.")
    else:
        print("✅ Base de documents existante chargée.")

    # 4. Création de la chaîne RAG
    retrieval_chain = create_rag_chain(retriever)

    print("✅ Système RAG prêt !")
    return retrieval_chain, retriever


def main():
    """Interface interactive pour poser des questions"""
    try:
        rag_system = setup_rag_system()
        chat_history = [] 

        print("\n" + "="*50)
        print("🤖 Système RAG Amélioré (Mémoire + Persistance)")
        print("="*50)
        print("Posez vos questions sur le document.")
        print("Tapez 'quit' pour quitter.")
        print("Tapez 'clear' pour effacer l'historique.")
        print("="*50)

        while True:
            question = input("\n❓ Votre question: ").strip()

            if question.lower() == 'quit':
                print("👋 Au revoir !")
                break
            
            elif question.lower() == 'clear':
                chat_history = []
                print("🧹 Historique effacé.")
                continue

            elif question == '':
                continue

            try:
                print("🔍 Recherche en cours...")
                
                # Invocation avec l'historique
                response = rag_system.invoke({
                    "input": question,
                    "chat_history": chat_history
                })

                answer = response['answer']
                print(f"\n💡 Réponse: {answer}")

                # Mise à jour de l'historique
                chat_history.append(HumanMessage(content=question))
                chat_history.append(AIMessage(content=answer))

                for doc in response['context']:
                    print(f"[CHUNK]: {doc.page_content}\n")

            except Exception as e:
                print(f"❌ Erreur lors de la recherche: {e}")
                print("Vérifiez qu'Ollama est bien démarré.")

    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")

if __name__ == "__main__":
    main()