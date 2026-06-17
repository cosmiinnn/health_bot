from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings # <-- Importăm Google
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Încărcăm cheia API pentru ca modelul de embeddings să o poată folosi
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Te rog să pui GOOGLE_API_KEY în fișierul .env!")

print("Citesc fișierele din folderele medicale...")

# 1. Încărcăm datele (ajustează numele folderelor dacă e cazul)
loader_afectiuni = DirectoryLoader('./date_medicale_afectiuni', glob="./*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
doc_afectiuni = loader_afectiuni.load()

# Dacă nu mai ai folderul de analize, poți șterge aceste 2 linii și la linia 24 lași doar documente = doc_afectiuni
loader_analize = DirectoryLoader('./date_medicale_analize', glob="./*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
doc_analize = loader_analize.load()

documente = doc_afectiuni + doc_analize
print(f"Am citit {len(documente)} documente în total.")

# 2. Spargem textul
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
bucati_text = text_splitter.split_documents(documente)
print(f"S-au generat {len(bucati_text)} fragmente de text.")

# 3. Creăm Embeddings cu Google (mult mai performant pentru limba română!)
print("Inițializez modelul Google Generative AI Embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

print("Construiesc noua bază de date FAISS...")
vector_db = FAISS.from_documents(bucati_text, embeddings)

# Salvăm într-un FOLDER NOU ca să nu îl stricăm pe cel vechi de la HuggingFace
vector_db.save_local("db_medicala_google")
print("✅ Baza de date Google a fost creată și salvată cu succes (db_medicala_google)!")