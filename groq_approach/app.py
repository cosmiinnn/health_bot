import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate # <-- Import nou!

# --- 1. INCARCARE AUTOMATA A VARIABILELOR DE MEDIU (.env) ---
load_dotenv()
api_key_env = os.getenv("GROQ_API_KEY")

# --- 2. CONFIGURARE INTERFAȚĂ STREAMLIT ---
st.set_page_config(page_title="Health Bot v2.1", page_icon="⚕️", layout="wide")
st.title("⚕️ Asistent Conversațional Medical")
st.markdown("*Sistem RAG bazat pe Llama 3.1/3.3 și FAISS*")

# --- 3. CACHING PENTRU BAZA DE DATE ---
@st.cache_resource
def incarca_baza_de_date():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("db_medicala_faiss", embeddings, allow_dangerous_deserialization=True)
    return vector_db

db = incarca_baza_de_date()

prompt_medical_template = """Ești un asistent medical virtual. 
Folosește "Contextul Extras" pentru a răspunde la întrebare. 

1. Dacă informația din context este utilă, bazează-te pe ea.
2. Dacă întrebarea este despre o afecțiune (ex: AVC), definește-o clar, prezintă simptomele și eventualele opțiuni generale.
3. Dacă contextul extras nu are absolut nicio legătură cu întrebarea, folosește-ți cunoștințele generale pentru a oferi un răspuns educativ, dar menționează scurt la început: "Nu am găsit un document specific, dar iată informațiile generale:"
4. Fii concis, clar și folosește liste.
5. Adaugă mereu la final: "Atenție: Informațiile sunt pur orientative și nu înlocuiesc un consult medical de specialitate."

Context Extras:
{context}

Întrebarea Utilizatorului: {question}

Răspuns:"""

PROMPT_MEDICAL = PromptTemplate(
    template=prompt_medical_template, 
    input_variables=["context", "question"]
)

# --- PROMPT PENTRU REFORMULAREA ÎNTREBĂRII ---
condense_template = """Analizează istoricul conversației și noua întrebare a utilizatorului. 
Reformulează noua întrebare pentru a fi clară și independentă.
REGULĂ CRITICĂ: Dacă utilizatorul folosește abrevieri medicale (ex: AVC, RMN, EKG, HTA), ești OBLIGAT să scrii cuvintele întregi în întrebarea reformulată (ex: Accident Vascular Cerebral, Tensiune Arterială).

Istoric:
{chat_history}

Întrebare nouă: {question}

Întrebare reformulată:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

# --- 5. BARA LATERALĂ (SETĂRI ȘI MODELE) ---
with st.sidebar:
    st.header("⚙️ Configurare")
    
    modele_disponibile = {
        "Llama 3.1 8B (Rapid)": "llama-3.1-8b-instant",
        "Llama 3.3 70B (Inteligent)": "llama-3.3-70b-versatile"
    }
    nume_model_selectat = st.selectbox("Alege creierul asistentului:", list(modele_disponibile.keys()))
    model_id = modele_disponibile[nume_model_selectat]
    
    if not api_key_env:
        st.warning("⚠️ Nu am găsit cheia în .env")
        api_key_manual = st.text_input("Introdu Groq API Key manual:", type="password")
    else:
        st.success("✅ Cheia API a fost încărcată.")
        api_key_manual = api_key_env

    if st.button("🗑️ Șterge Conversația"):
        st.session_state.mesaje = []
        st.session_state.memorie.clear()
        st.rerun()

# --- 6. INIȚIALIZARE MEMORIE CHAT ---
if "mesaje" not in st.session_state:
    st.session_state.mesaje = []
    
if "memorie" not in st.session_state:
    st.session_state.memorie = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

for mesaj in st.session_state.mesaje:
    with st.chat_message(mesaj["rol"]):
        st.markdown(mesaj["continut"])

# --- 7. LOGICA DE CONVERSAȚIE ---
intrebare = st.chat_input("Adresează o întrebare medicală...")

if intrebare:
    if not api_key_manual:
        st.error("Te rog să configurezi Groq API Key!")
        st.stop()

    with st.chat_message("user"):
        st.markdown(intrebare)
    st.session_state.mesaje.append({"rol": "user", "continut": intrebare})

    # Setăm temperatura la 0.1 - un echilibru bun
    llm = ChatGroq(
        groq_api_key=api_key_manual, 
        model_name=model_id, 
        temperature=0.1 
    )

    lant_conversatie = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 4}), # Am scos MMR-ul!
        memory=st.session_state.memorie,
        combine_docs_chain_kwargs={"prompt": PROMPT_MEDICAL},
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        verbose=True
    )

    with st.chat_message("assistant"):
        with st.spinner(f"Se procesează folosind {nume_model_selectat}..."):
            try:
                raspuns_bot = lant_conversatie.invoke({"question": intrebare})
                text_raspuns = raspuns_bot["answer"]
                st.markdown(text_raspuns)
                st.session_state.mesaje.append({"rol": "assistant", "continut": text_raspuns})
            except Exception as e:
                st.error(f"Eroare API: {str(e)}")