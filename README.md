
# 🧬 BioMedical RAG — Assistant Étudiant Local

**Python · Docker · Qdrant · BioMistral-7B (LlamaCpp) · PubMedBERT · FastAPI**

Assistant de questions-réponses **hors-ligne** sur des **PDF médicaux** (conçu pour les étudiants de la FMPR Rabat).
Pipeline **RAG** : *Embeddings PubMedBERT → Qdrant (vector DB) → BioMistral-7B via LlamaCpp → FastAPI + UI Bootstrap.*

---

## ✨ Objectif

Aider les **étudiants** à :

* trouver rapidement le **passage pertinent** dans de grands PDF,
* obtenir une **réponse expliquée**,
* garder les **données en local** (confidentialité).

---

## 🧱 Architecture

```
PDFs
  └─ Loader (Unstructured/DirectoryLoader)
       └─ Splitter (RecursiveCharacterTextSplitter 700/70)
            └─ Embeddings (PubMedBERT)
                 └─ Qdrant (vector_db, cosine)
                      └─ Retriever (k=1)
                           └─ LLM (BioMistral-7B via LlamaCpp)
                                └─ Réponse (+ passage source)
```

---

## 🧰 Stack technique

* **Langage & Conteneur** : Python, Docker
* **IA & NLP (RAG)** : LlamaCpp (BioMistral-7B.Q4_K_M.gguf), PubMedBERT embeddings, LangChain (PromptTemplate, RetrievalQA)
* **Vector DB** : Qdrant (`qdrant_client`, `langchain_community.vectorstores.Qdrant`)
* **Web** : FastAPI, Jinja2Templates, Bootstrap 5
* **I/O** : UnstructuredFileLoader, DirectoryLoader, RecursiveCharacterTextSplitter

---

## 📁 Arborescence (simplifiée)

```
.
├─ app.py                     # API FastAPI + routes (chat, upload, index, delete)
├─ create_vector_db.py        # script d’indexation (ou notebook .ipynb)
├─ settings.py                # configuration centrale du projet
├─ requirements.txt
├─ BioMistral-7B.Q4_K_M.gguf  # (optionnel si chargé ailleurs — voir settings.LLM_PATH)
├─ data/                      # déposer ici vos PDF
├─ templates/
│  └─ index.html              # UI (upload + chat)
├─ static/                    # CSS / JS / images
└─ guide.ipynb                # pense-bête (docker qdrant, uvicorn)
```

---

## ⚙️ Configuration (fichier `settings.py`)

```python
VECTOR_DB_URL = "http://localhost:6333"
VECTOR_DB_NAME = "vector_db"
DATA_DIR = "data/"
EMBEDDINGS = "NeuML/pubmedbert-base-embeddings"  # ou chemin local
LLM_PATH = "BioMistral-7B.Q4_K_M.gguf"           # chemin vers le .gguf
PROMPT_TEMPLATE = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""
```

> Vous pouvez pointer `EMBEDDINGS` et `LLM_PATH` vers des **dossiers/fichiers locaux** (voir section “Télécharger depuis Hugging Face”).

---

## 🧩 Prérequis

* Python 3.10+ (OK avec 3.11)
* Docker (pour Qdrant)
* Accès Internet **uniquement** pour télécharger les modèles (usage ensuite possible **offline**)

---

## 📥 Télécharger les modèles depuis Hugging Face

> Prérequis : `pip install -U huggingface_hub`
> (Si le repo est “gated”, exécutez `huggingface-cli login`.)

### Embeddings — PubMedBERT

```bash
huggingface-cli download NeuML/pubmedbert-base-embeddings \
  --local-dir ./models/pubmedbert-base-embeddings
```

Dans `settings.py` :

```python
EMBEDDINGS = "./models/pubmedbert-base-embeddings"
```

*(Vous pouvez aussi garder l’ID HF : LangChain téléchargera automatiquement.)*

### LLM — BioMistral-7B (format GGUF pour LlamaCpp)

```bash
# Adapter <ORG>/<REPO> et le nom exact du fichier .gguf selon le dépôt choisi
huggingface-cli download <ORG>/<REPO> <FICHIER_GGUF> \
  --local-dir ./models/biomistral
```

Dans `settings.py` :

```python
LLM_PATH = "./models/biomistral/BioMistral-7B.Q4_K_M.gguf"
```

### Astuce Docker (volumes)

```bash
-v $(pwd)/models:/models
# settings.py
LLM_PATH = "/models/biomistral/BioMistral-7B.Q4_K_M.gguf"
EMBEDDINGS = "/models/pubmedbert-base-embeddings"
```

---

## 🛠️ Installation

```bash
# 1) Environnement
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 2) Dépendances Python
pip install -r requirements.txt
# (si besoin)
pip install -U huggingface_hub
```

---

## 🐳 Lancer Qdrant (Docker)

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 --rm qdrant/qdrant
```

---

## 🧮 Créer l’index vectoriel

Assurez-vous que vos **PDF** sont dans `data/`.

**Option A — Script**

```bash
python create_vector_db.py
```

**Option B — Notebook**
Ouvrir `create_vector_db.ipynb` et exécuter.

> Paramètres par défaut : `chunk_size=700`, `chunk_overlap=70`, distance **Cosine**, collection **vector_db**.

---

## ▶️ Démarrer l’application

```bash
uvicorn app:app --reload
```

* UI : [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Page web avec : **upload document**, **upload all**, **delete all**, zone de **chat**.

---

## 🔌 Endpoints principaux

* `GET /` → page HTML (UI)
* `POST /get_response` (form-data: `query`)
  → `{"answer","source_document","doc"}`
* `POST /upload_document` (form-data: `file` .pdf)
  → upload + indexation
* `POST /upload_all_documents`
  → indexe tous les PDF de `data/`
* `POST /sup_all`
  → supprime la collection Qdrant

### Exemples `curl`

```bash
# Poser une question
curl -X POST -F "query=Quel est le protocole X ?" http://127.0.0.1:8000/get_response

# Uploader un PDF
curl -X POST -F "file=@/chemin/vers/cours.pdf" http://127.0.0.1:8000/upload_document
```

---

## 🧪 Réglages utiles

* **Retriever** : `db.as_retriever(search_kwargs={"k": 1})`
  (Augmentez `k` si vous voulez plus de contexte)
* **Prompt** : modifiez `PROMPT_TEMPLATE` dans `settings.py`
* **Nettoyage** : `POST /sup_all` supprime la collection Qdrant

---

## 🛟 Dépannage

* **Qdrant inaccessible** : Docker lancé ? Port `6333` libre ?
* **Modèle introuvable** : vérifiez `LLM_PATH` et le nom exact du `.gguf`.
* **Lent sur CPU** : le modèle est quantifié (**Q4_K_M**), mais 7B reste lourd. Fermez les applis lourdes.
* **Erreur Pydantic/FastAPI** : alignez les versions

  ```bash
  pip install "fastapi>=0.110" "pydantic>=2" uvicorn
  ```
* **Upload refusé** : seul **.pdf** est accepté par l’endpoint d’upload.

---

## 📜 Licence & crédits

Respectez les licences de : **BioMistral-7B**, **PubMedBERT**, **Qdrant**, **LangChain**, **FastAPI**, **Bootstrap**.
Projet académique visant à **faciliter les révisions des étudiants**.

---

