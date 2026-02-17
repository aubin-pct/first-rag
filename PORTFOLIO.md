# First-RAG : Architecture RAG Avancée & Apprentissage

Ce projet est une implémentation complète d'un système de **Retrieval-Augmented Generation (RAG)**. Il représente une **démarche d'apprentissage approfondie** visant à maîtriser les concepts clés de l'IA générative appliquée aux documents, au-delà des simples tutoriels.


## Objectifs & Apprentissage

L'ambition principale de ce projet était de déconstruire et comprendre chaque composant d'un pipeline RAG moderne, plutôt que d'utiliser des solutions "clés en main" abstraites.

### Concepts Clés Maîtrisés
À travers ce développement, j'ai exploré et implémenté les concepts suivants :
*   **Vector Embeddings & Semantic Search** : Compréhension de la projection de texte en vecteurs denses (via `Sentence Transformers`) pour capturer le sens au-delà des simples mots-clés. J'ai appris comment les embeddings transforment la sémantique en distance mathématique (Similarité cosinus).
*   **Stratégies de Chunking Avancées** : Apprentissage des nuances entre un découpage naïf (par caractères) et le **Chunking Sémantique** (découper dynamiquement là où le sens change), essentiel pour ne pas tronquer les idées.
*   **Indexation Avancée (Parent-Child)** : Mise en œuvre du pattern **Parent-Child Indexing** pour découpler "ce qu'on cherche" (petits fragments précis pour le vecteur) de "ce qu'on donne au LLM" (bloc parent complet pour le contexte). Cela résout le problème de perte de contexte fréquent dans les RAG simples.
*   **Reranking & Cross-Encoders** : Comprendre le phénomène "Lost in the Middle" et pourquoi la recherche vectorielle bi-encoder manque parfois de précision fine. L'intégration d'un **Cross-Encoder** (BGE) m'a permis de réévaluer la pertinence réelle des documents retrouvés avant l'étape de génération.
*   **Architecture Modulaire & Design Patterns** : Conception d'un pipeline flexible (Router, Loader, Retriever) permettant de changer de composants (ex: passer de ChromaDB à PGVector) sans refondre le système.


## Choix Techniques & Alternatives

Un point clé de l'apprentissage a été l'arbitrage constant entre performance technique, coût d'infrastructure et complexité de mise en œuvre.

### 1. Vision et Parsing de Documents : Le Défi des Tableaux
*   **Choix retenu : LlamaParse**
    *   *Pourquoi ?* C'est une solution spécialisée qui reconstruit la structure des documents (tableaux, titres) en Markdown. Cela permet au LLM de "comprendre" la structure spatiale des données sans voir l'image.
*   **Alternative envisagée : LLM Multimodaux Vision**
    *   *Concept* : Envoyer directement les images des pages PDF au LLM pour qu'il "lise" visuellement le document.
    *   *Pourquoi pas ?* Bien que très performant, le **coût aurait explosé** pour de gros volumes documentaires. Un PDF de 100 pages traité page par page en vision est extrêmement coûteux en tokens. LlamaParse offre un compromis "one-shot" beaucoup plus économique et suffisant pour mon cas.

### 2. Modèle de Langage (Inférence)
*   **Développement : Ollama (Local)**
    *   *Pourquoi ?* J'ai utilisé **Ollama** pour faire tourner les modèles (Llama 3) entièrement en local pendant la phase de développement. Cela m'a permis d'itérer rapidement sans coût, sans latence réseau, et en gardant la maîtrise totale de l'infrastructure.
*   **Déploiement Portfolio : Groq (Cloud)**
    *   *Pourquoi ?* Pour la démonstration publique, j'ai basculé sur l'API **Groq**. Son offre gratuite généreuse et sa vitesse d'inférence vous permet de tester le RAG avec une fluidité, sans que j'aie à héberger un serveur GPU coûteux.

### 3. Base de Données Vectorielle
*   **Choix retenu : ChromaDB (Local)**
    *   *Pourquoi ?* Simplicité de mise en place (intégré), persistance locale sans Docker lourd -> prototypage rapide.


## L'Architecture Implémentée

J'ai conçu une architecture modulaire pour répondre aux problèmes courants des RAG basiques.

### 1. Ingestion Intelligente (Router & Parsing)
Le système utilise un **Router** pour diriger les fichiers :
*   **Pipeline Vision** : Traite les documents riches (tableaux, mises en page) via LlamaParse.
*   **Pipeline Texte** : Traite les textes simples via PyPDF pour la rapidité.

### 2. Recherche Hybride & Reranking
Pour pallier les faiblesses de la recherche sémantique (manque de précision sur les termes techniques) :
1.  **Ensemble Retriever** : Combine **BM25** (mots-clés) + **ChromaDB** (vecteurs).
2.  **Reranker (BGE)** : Ré-ordonne les résultats pour placer les plus pertinents en premier.

### 3. Chat & Mémoire
*   Gestion de l'historique de conversation (Stateful) via **PostgreSQL** (Dockerisé).
*   Reformulation contextuelle des questions ("C'est quoi ?" devient "C'est quoi [le sujet précédent] ?").


## 🛠 Stack Technique

**Backend & API**
*   **Python 3.10+** & **FastAPI** : Pour une API asynchrone robuste.
*   **PostgreSQL** : Base de données relationnelle (via Docker) pour la persistance des sessions.
*   **SQLAlchemy** : ORM pour l'interaction avec la BDD.

**Intelligence Artificielle**
*   **LangChain** : Framework d'orchestration.
*   **ChromaDB** : Stockage vectoriel.
*   **HuggingFace Embeddings** : `all-MiniLM-L6-v2` (bon ratio performance/vitesse).
*   **BAAI/bge-reranker** : Pour le reranking de précision.

**Outils**
*   **Docker** : Conteneurisation.
*   **LlamaParse** : OCR intelligent.
