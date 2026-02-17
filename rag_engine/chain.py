from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import LLM_MODEL, GROQ_API_KEY

def create_rag_chain(retriever):
    """
    Crée une chaîne RAG avec gestion de l'historique de conversation.
    
    Streaming: Mode de transmission où la réponse du modèle est envoyée morceau par morceau (token par token) dès qu'elle est générée, permettant un affichage progressif et plus réactif pour l'utilisateur.
    """
    # Activation du streaming pour une meilleure réactivité (si supporté par l'interface)
    llm = ChatGroq(model=LLM_MODEL, temperature=0.1, streaming=True, api_key=GROQ_API_KEY)

    # 1. Chaîne pour reformuler la question en fonction de l'historique
    contextualize_q_system_prompt = """Compte tenu de l'historique de la conversation et de la dernière question de l'utilisateur 
    qui pourrait faire référence au contexte de l'historique de la conversation, formulez une question autonome 
    qui peut être comprise sans l'historique de la conversation. Ne répondez pas à la question, 
    reformulez-la simplement si nécessaire, sinon renvoyez-la telle quelle."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Chaîne pour répondre à la question (QA)
    qa_system_prompt = """Tu es un assistant expert en analyse de documents.
    Utilise les morceaux de contexte récupérés suivants pour répondre à la question.
    Si tu ne connais pas la réponse, dis simplement que tu ne sais pas.
    Utilise trois phrases maximum et sois concis.

    Contexte:
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 3. Chaîne finale combinant les deux
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain