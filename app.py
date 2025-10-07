# Importation des bibliothèques nécessaires
from langchain import PromptTemplate  # Gestion des prompts pour le LLM
from langchain_community.llms import LlamaCpp  # Modèle Llama pour la génération de texte
from langchain.chains import RetrievalQA  # Chaîne pour les questions-réponses avec récupération
from langchain_community.embeddings import SentenceTransformerEmbeddings  # Génération d'embeddings
from fastapi import FastAPI, Request, Form, Response, UploadFile, HTTPException  # Framework pour créer une API
from fastapi.responses import HTMLResponse  # Réponse HTML pour les pages
from fastapi.templating import Jinja2Templates  # Gestion des templates HTML
from fastapi.staticfiles import StaticFiles  # Fichiers statiques (CSS, JS, images)
from fastapi.encoders import jsonable_encoder  # Conversion des données en JSON
from qdrant_client import QdrantClient, models  # Client pour interagir avec Qdrant
from langchain_community.vectorstores import Qdrant  # Interface avec Qdrant
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader  # Chargement des documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Découpe des documents en morceaux
import settings  # Fichier de configuration
import json  # Bibliothèque pour manipuler les données JSON
import os  # Bibliothèque pour manipuler les fichiers



# Initialisation de l'application FastAPI
app = FastAPI()

# Configuration des templates et des fichiers statiques
templates = Jinja2Templates(directory="templates")  # Répertoire contenant les fichiers HTML
app.mount("/static", StaticFiles(directory="static"), name="static")  # Répertoire pour les fichiers statiques

# Initialisation du modèle local Llama avec les paramètres spécifiés
local_llm = settings.LLM_PATH
llm = LlamaCpp(
    model_path=local_llm,  # Chemin vers le modèle Llama
    temperature=0.3,  # Contrôle la variabilité des réponses
    max_tokens=2048,  # Nombre maximal de tokens générés
    top_p=1  # Contrôle la diversité des réponses
)
print("LLM Initialized....")  # Confirmation de l'initialisation du modèle

# Initialisation des embeddings
embeddings = SentenceTransformerEmbeddings(model_name=settings.EMBEDDINGS)

# Connexion à la base de données Qdrant
client = QdrantClient(url=settings.VECTOR_DB_URL, prefer_grpc=False)  # Client pour interagir avec Qdrant

# Vérifier si la collection existe, sinon la créer
try:
    collections = client.get_collections().collections
    if not any(collection.name == settings.VECTOR_DB_NAME for collection in collections):
        client.create_collection(
            collection_name=settings.VECTOR_DB_NAME,
            vectors_config=models.VectorParams(size=embeddings.dim, distance="Cosine")
        )
        print(f"Collection '{settings.VECTOR_DB_NAME}' created.")
    else:
        print(f"Collection '{settings.VECTOR_DB_NAME}' already exists.")
except Exception as e:
    print(f"Error checking or creating collection: {e}")

db = Qdrant(client=client, embeddings=embeddings, collection_name=settings.VECTOR_DB_NAME)  # Chargement de la collection

# Configuration du modèle de prompt
prompt = PromptTemplate(template=settings.PROMPT_TEMPLATE, input_variables=['context', 'question'])

# Définition du retriever (outil pour récupérer les documents pertinents)
retriever = db.as_retriever(search_kwargs={"k": 1})  # Recherche les 1 document(s) les plus pertinents

# Route pour la page d'accueil
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})  # Retourne le fichier HTML principal

# Route pour gérer les requêtes utilisateur
@app.post("/get_response")
async def get_response(query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}  # Configure le prompt pour la chaîne
    qa = RetrievalQA.from_chain_type(
        llm=llm,  # Modèle de génération de texte
        chain_type="stuff",  # Type de chaîne
        retriever=retriever,  # Méthode de récupération des documents
        return_source_documents=True,  # Retourne les documents sources
        chain_type_kwargs=chain_type_kwargs,
        verbose=True  # Active les logs détaillés
    )
    response = qa(query)  # Obtenez la réponse à la requête
    print(response)  # Affiche la réponse pour vérification

    # Extraction des informations de la réponse
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']

    # Encodage des données en JSON
    response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))
    res = Response(response_data)  # Prépare une réponse HTTP
    return res  # Retourne la réponse

# Route pour charger un document dans le dossier data et l'indexer
@app.post("/upload_document")
async def upload_document(file: UploadFile):
    try:
        # Assurez-vous que le dossier existe
        os.makedirs(settings.DATA_DIR, exist_ok=True)

        # Enregistrer le fichier dans le dossier `data`
        file_location = os.path.join(settings.DATA_DIR, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Vérifier le type de fichier
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        # Charger et indexer le fichier
        loader = UnstructuredFileLoader(file_location)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
        texts = text_splitter.split_documents(document)
        qdrant = Qdrant.from_documents(
            texts,  # Textes découpés à indexer
            embeddings,  # Modèle d'embeddings pour convertir les textes en vecteurs
            url=settings.VECTOR_DB_URL,  # URL de la base de données Qdrant
            prefer_grpc=False,  # Préfère HTTP au lieu de gRPC
            collection_name=settings.VECTOR_DB_NAME  # Nom de la collection à créer
        )

        return {"message": f"Document '{file.filename}' uploaded to '{settings.DATA_DIR}' and indexed successfully!"}
    except Exception as e:
        # Capture l'erreur et retourne un message clair
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Route pour charger tous les documents d'un répertoire dans la base de données
@app.post("/upload_all_documents")
async def upload_all_documents():
    try:




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

        return {"message": "All documents from the directory were uploaded and indexed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while uploading all documents: {str(e)}")



# Route pour charger tous les documents d'un répertoire dans la base de données
@app.post("/sup_all")
async def sup_all():
    try:
                # Connexion à la base de données vectorielle Qdrant
        client = QdrantClient(url=settings.VECTOR_DB_URL, prefer_grpc=False)

        # Supprime tout le contenu de la collection spécifiée
        def clear_collection(collection_name):
            try:
                client.delete_collection(collection_name)
                print(f"La collection '{collection_name}' a été supprimée avec succès.")
            except Exception as e:
                print(f"Erreur lors de la suppression de la collection : {e}")

        # Appel de la fonction pour supprimer le contenu de la collection
        clear_collection(settings.VECTOR_DB_NAME)




        return {"message": "All documents from the directory were uploaded and indexed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while uploading all documents: {str(e)}")
