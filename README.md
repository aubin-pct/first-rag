# 🧠 Système RAG (Retrieval-Augmented Generation)

## Vue d'ensemble

Ce système RAG permet d'interroger des documents (PDF, texte) via une interface de chat. Il combine la recherche d'information avec la génération de réponses par un modèle de langage (LLM).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ARCHITECTURE RAG                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   📄 Documents                                                              │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────┐     ┌──────────────────────────────────────────────┐     │
│   │   ROUTER    │────▶│  Pipeline Vision (LlamaParse)                │     │
│   │  (Gardien)  │     │  - OCR avancé, tables, images                │     │
│   └─────────────┘     └──────────────────────────────────────────────┘     │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  Pipeline Texte (PyPDF)                                          │     │
│   │  - Extraction rapide, chunking récursif                          │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │              INDEXATION PARENT-CHILD                             │     │
│   │  ┌─────────────────┐       ┌─────────────────┐                   │     │
│   │  │   DocStore      │       │   VectorStore   │                   │     │
│   │  │   (Parents)     │◄─────▶│   (Enfants)     │                   │     │
│   │  │   Docs complets │       │   Petits chunks │                   │     │
│   │  └─────────────────┘       └─────────────────┘                   │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│                            RECHERCHE                                        │
│                               │                                             │
│        ┌──────────────────────┼──────────────────────┐                     │
│        ▼                      ▼                      ▼                     │
│   ┌─────────┐          ┌─────────────┐        ┌──────────┐                 │
│   │  BM25   │          │  Vectoriel  │        │ Reranker │                 │
│   │ (Mots)  │          │ (Sémantique)│───────▶│   BGE    │                 │
│   └─────────┘          └─────────────┘        └──────────┘                 │
│        │                      │                      │                     │
│        └──────────────────────┴──────────────────────┘                     │
│                               │                                             │
│                               ▼                                             │
│                    ┌─────────────────────┐                                  │
│                    │  EnsembleRetriever  │                                  │
│                    │   (40% BM25 +       │                                  │
│                    │    60% Vectoriel)   │                                  │
│                    └─────────────────────┘                                  │
│                               │                                             │
│                               ▼                                             │
│                    ┌─────────────────────┐                                  │
│                    │      LLM (Ollama)   │                                  │
│                    │   llama3.1:8b       │                                  │
│                    └─────────────────────┘                                  │
│                               │                                             │
│                               ▼                                             │
│                         💬 Réponse                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Structure des fichiers

```
lib/rag/
├── config.py              # Configuration globale (modèles, chemins, paramètres)
├── main.py                # API FastAPI (endpoints /chat, /sessions)
├── database.py            # Connexion PostgreSQL (SQLAlchemy)
├── models.py              # Modèles DB (ChatSession, ChatMessage)
├── schemas.py             # Schémas Pydantic pour l'API
├── data/                  # 📂 Placez vos documents ici (PDF, TXT)
└── rag_engine/
    ├── service.py         # Point d'entrée du système RAG
    ├── loader.py          # Chargement des documents via le Router
    ├── router.py          # Routage intelligent vers le bon pipeline
    ├── vector_store.py    # Gestion des stores (Chroma, DocStore, Retrievers)
    ├── chain.py           # Création de la chaîne LangChain
    ├── reranker.py        # Compresseur BGE pour le reranking
    └── pipelines/
        ├── base.py        # Interface de base pour les pipelines
        ├── text_pipeline.py   # Pipeline pour documents textuels
        └── vision_pipeline.py # Pipeline pour documents complexes (LlamaParse)
```

---

## 🔧 Composants techniques

### 1. Ingestion Intelligente (Pipelines)

Le système analyse chaque document et choisit automatiquement le meilleur pipeline de traitement.

| Pipeline | Usage | Technologie |
|----------|-------|-------------|
| **Vision** | PDF complexes (tables, images, mise en page) | LlamaParse (API Cloud) |
| **Texte** | Fichiers texte, PDF simples | PyPDFLoader + Chunking récursif |

**Router** : Composant logiciel qui analyse les métadonnées d'un fichier pour diriger son traitement vers le pipeline approprié.

### 2. Indexation Parent-Child

Cette stratégie permet d'avoir le meilleur des deux mondes :
- **Recherche précise** : Les petits chunks (enfants, 400 tokens) permettent une correspondance fine avec la requête.
- **Contexte riche** : Les documents complets (parents) sont retournés au LLM pour une réponse de qualité.

| Store | Contenu | Format |
|-------|---------|--------|
| **VectorStore (Chroma)** | Chunks enfants vectorisés | Vecteurs (Embeddings) |
| **DocStore (LocalFileStore)** | Documents parents complets | Pickle (sérialisé) |

**Embedding** : Processus de conversion d'un texte en un vecteur numérique de dimension fixe, capturant son sens sémantique.

### 3. Recherche Hybride

La recherche combine deux approches complémentaires :

| Méthode | Force | Poids |
|---------|-------|-------|
| **BM25** | Correspondance exacte de mots-clés (TF-IDF amélioré) | 40% |
| **Vectoriel** | Compréhension sémantique (sens proche) | 60% |

**BM25** : Algorithme probabiliste de recherche d'information basé sur la fréquence des mots.

### 4. Reranking

Après la recherche initiale, un modèle de Deep Learning réévalue et réordonne les résultats.

- **Modèle** : `BAAI/bge-reranker-v2-m3`
- **Processus** : Les chunks enfants sont rerankés, puis les parents uniques correspondants sont récupérés.

**Reranker** : Modèle spécialisé qui attribue un score de pertinence à chaque paire (requête, document).

### 5. Caching & Optimisation

| Cache | Utilité | Stockage |
|-------|---------|----------|
| **LLM Cache** | Évite de rappeler le LLM pour des questions identiques | SQLite (`cache/llm_cache.db`) |
| **Embeddings Cache** | Évite de recalculer les vecteurs déjà connus | Fichiers (`cache/embeddings_cache/`) |

---

## ⚙️ Configuration (`config.py`)

```python
# Modèles
LLM_MODEL = "llama3.1:8b"          # Modèle de génération (Ollama)
EMBEDDING_MODEL = "nomic-embed-text" # Modèle d'embeddings (Ollama)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3" # Modèle de reranking (HuggingFace)

# Chunking
CHUNK_SIZE = 1000        # Taille des Parents
CHILD_CHUNK_SIZE = 400   # Taille des Enfants (recherche vectorielle)

# Recherche
SEARCH_K = 10            # Nombre de résultats à retourner
USE_RERANKER = True      # Activer le reranking BGE
USE_HYBRID_SEARCH = True # Activer BM25 + Vectoriel
```

---

## 🚀 API Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/chat` | Envoyer une question et recevoir une réponse |
| `GET` | `/sessions` | Liste des conversations (triées par épinglage puis date) |
| `GET` | `/sessions/{id}/messages` | Messages d'une conversation |
| `DELETE` | `/sessions/{id}` | Supprimer une conversation |
| `PATCH` | `/sessions/{id}/pin` | Épingler/désépingler une conversation |

### Exemple de requête `/chat`

```json
{
  "question": "Quel est le risque de ce fonds ?",
  "history": [
    {"role": "user", "content": "Bonjour"},
    {"role": "assistant", "content": "Bonjour ! Comment puis-je vous aider ?"}
  ],
  "session_id": 1
}
```

---

## 📊 Glossaire technique

| Terme | Définition |
|-------|------------|
| **RAG** | Retrieval-Augmented Generation : Technique d'IA qui améliore les réponses d'un LLM en lui fournissant des informations pertinentes récupérées dans une base de connaissances externe. |
| **LLM** | Large Language Model : Modèle de langage de grande taille capable de générer du texte. |
| **Chunking** | Découpage d'un long texte en segments plus courts pour faciliter leur indexation. |
| **Vector Store** | Base de données optimisée pour stocker et rechercher des vecteurs. |
| **DocStore** | Système de stockage clé-valeur conservant les documents originaux complets. |
| **Retriever** | Composant chargé de retrouver les documents les plus pertinents. |
| **Pipeline** | Chaîne de traitement séquentielle (Chargement → Nettoyage → Découpage). |
| **Streaming** | Mode de transmission où la réponse est envoyée progressivement (token par token). |

---

## 🔄 Flux de traitement d'une question

```
1. Question utilisateur
       │
       ▼
2. Reformulation (si historique de conversation)
       │
       ▼
3. Recherche hybride (BM25 + Vectoriel)
       │
       ▼
4. Reranking des chunks enfants
       │
       ▼
5. Récupération des documents parents
       │
       ▼
6. Génération de la réponse par le LLM
       │
       ▼
7. Sauvegarde en base de données
       │
       ▼
8. Réponse à l'utilisateur
```

---

## 🛠️ Prérequis

### 1. Installation d'Ollama

Ollama permet d'exécuter des modèles de langage localement sur votre machine.

#### macOS (avec Homebrew)
```bash
brew install ollama
```

#### Autres systèmes
Téléchargez Ollama depuis [ollama.ai](https://ollama.ai) et suivez les instructions d'installation.

### 2. Installation des modèles

Après avoir installé Ollama, téléchargez les modèles nécessaires :

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 3. Démarrage du serveur Ollama

Lancez le serveur Ollama :

```bash
ollama serve
```

### 4. Docker (PostgreSQL)

Pour le stockage des sessions de chat :

```bash
docker-compose up -d
```

### 5. Installation des dépendances Python

```bash
pip install langchain-ollama langchain-chroma langchain-community pypdf llama-parse python-dotenv fastapi uvicorn sqlalchemy psycopg2-binary sentence-transformers
```

### 6. Configuration LlamaParse (Optionnel)

Pour une meilleure lecture des PDF (tableaux, mises en page complexes), créez un fichier `.env` :

```bash
LLAMA_CLOUD_API_KEY=votre_clé_api
```

Obtenez une clé API gratuite sur [LlamaCloud](https://cloud.llamaindex.ai/).

---

## 📝 Utilisation

1. Placez vos documents dans `lib/rag/data/`
2. Lancez l'application : `./run_app.sh`
3. L'indexation se fait automatiquement au premier démarrage
4. Posez vos questions via l'interface Flutter ou directement via l'API