from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

print("Citesc fișierele din folderele medicale...")

loader_afectiuni = DirectoryLoader('./date_medicale_afectiuni', glob="./*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
doc_afectiuni = loader_afectiuni.load()

try:
    loader_analize = DirectoryLoader('./date_medicale_analize', glob="./*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    doc_analize = loader_analize.load()
    documente = doc_afectiuni + doc_analize
except Exception:
    documente = doc_afectiuni

print(f"Am citit {len(documente)} documente în total.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
bucati_text = text_splitter.split_documents(documente)
print(f"S-au generat {len(bucati_text)} fragmente de text.")

print("Inițializez modelul local de Embeddings")
# Folosim un model optimizat pentru limba română
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

print("Construiesc baza de date FAISS...")
vector_db = FAISS.from_documents(bucati_text, embeddings)

vector_db.save_local("db_medicala_hibrid")
print("✅ Baza de date hibridă a fost creată cu succes (db_medicala_hibrid)!")