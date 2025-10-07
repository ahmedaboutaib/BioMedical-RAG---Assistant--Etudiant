import os  # Module pour gérer les interactions avec le système de fichiers
from langchain_community.embeddings import SentenceTransformerEmbeddings  # Utilisé pour générer des embeddings à partir de textes
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader  # Charge les documents depuis un dossier
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Découpe les documents en morceaux plus petits
from langchain_community.vectorstores import Qdrant  # Interface pour la base de données vectorielle Qdrant
import settings  # Fichier de configuration contenant les paramètres du projet

# Initialisation du générateur d'embeddings avec le modèle spécifié dans settings
embeddings = SentenceTransformerEmbeddings(model_name=settings.EMBEDDINGS)
print(embeddings)  # Vérification que les embeddings sont correctement initialisés

# Charge tous les fichiers PDF du répertoire spécifié
loader = DirectoryLoader(settings.DATA_DIR, glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
documents = loader.load()  # Charge tous les documents dans une liste

# Découpe les documents en morceaux de 700 caractères avec un chevauchement de 70 caractères
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
texts = text_splitter.split_documents(documents)  # Liste des morceaux de textes découpés

print(texts[1])  # Exemple d'un morceau découpé pour vérification

# Création de la base de données vectorielle en utilisant Qdrant
qdrant = Qdrant.from_documents(
    texts,  # Textes découpés à indexer
    embeddings,  # Modèle d'embeddings pour convertir les textes en vecteurs
    url=settings.VECTOR_DB_URL,  # URL de la base de données Qdrant
    prefer_grpc=False,  # Préfère HTTP au lieu de gRPC
    collection_name=settings.VECTOR_DB_NAME  # Nom de la collection à créer
)

print("Vector DB Successfully Created!")  # Confirmation de la création de la base vectorielle
