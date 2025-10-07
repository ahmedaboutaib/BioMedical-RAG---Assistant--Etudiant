# URL de la base de données vectorielle Qdrant
VECTOR_DB_URL = "http://localhost:6333"

# Nom de la collection dans Qdrant
VECTOR_DB_NAME = "vector_db"

# Répertoire contenant les documents à traiter
DATA_DIR = "data/"

# Chemin ou nom du modèle d'intégration utilisé pour créer des embeddings
EMBEDDINGS = "NeuML/pubmedbert-base-embeddings"

# Chemin vers le modèle LLM (Large Language Model) utilisé pour la génération de réponses
LLM_PATH = "BioMistral-7B.Q4_K_M.gguf"

# Modèle de prompt utilisé pour formuler des requêtes au LLM
PROMPT_TEMPLATE = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""
