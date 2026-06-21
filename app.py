import streamlit as st
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI      

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# CONFIGURARE MEDIU ȘI STREAMLIT ---
load_dotenv()
api_key_env = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Health Bot", page_icon="🧬", layout="wide")
st.title("🧬 Health Bot")

# BAZA DE DATE LOCALĂ ---
@st.cache_resource
def incarca_baza_de_date():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    vector_db = FAISS.load_local("db_medicala_hibrid", embeddings, allow_dangerous_deserialization=True)
    return vector_db

db = incarca_baza_de_date()

# PROMPTURI STRICTE ---
condense_template = """Analizează istoricul conversației și noua întrebare a utilizatorului. 
Reformulează noua întrebare pentru a fi clară și independentă, pregătită pentru căutarea în baza de date.

REGULI PENTRU REFORMULARE:
1. Extinde abrevierile medicale românești (ex: AVC -> Accident Vascular Cerebral).
2. Corectează și capitalizează numele de medicamente sau patologii (ex: "aflamil" devine "Aflamil", "paracetamol" devine "Paracetamol").
3. Păstrează intenția exactă a utilizatorului.

Istoric:
{chat_history}
Întrebare nouă: {question}
Întrebare reformulată:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

prompt_medical_template = """Ești un asistent medical virtual expert și empatic. 
Analizează "Contextul Extras" pentru a răspunde la întrebarea utilizatorului. 

REGULI DE COMPORTAMENT:
1. Dacă informația din context este utilă, folosește-o și structurează răspunsul cu liste (bullet points).
2. Dacă utilizatorul cere interpretări de analize care nu apar explicit în context, completează RĂSPUNSUL cu propriile tale cunoștințe medicale (ex: "Conform documentelor [...], iar din practica medicală valorile normale sunt...").
3. INTERZIS: Sub nicio formă nu menționa numele unor clinici private (ex: Regina Maria, MedLife etc.) sau numere de telefon (ex: 021 9268) prezente în context. Formulează recomandările la modul general (ex: "adresează-te unui laborator clinic" sau "sună la medic/112").
4. Adaugă mereu la finalul răspunsului: "Atenție: Informațiile sunt pur orientative și nu înlocuiesc un consult medical de specialitate."

Context Extras:
{context}

Întrebarea Utilizatorului: {question}

Răspuns:"""
PROMPT_MEDICAL = PromptTemplate(template=prompt_medical_template, input_variables=["context", "question"])

# INTERFAȚA LATERALĂ ---
with st.sidebar:
    st.header("⚙️ Configurare model")
    
    modele_disponibile = {
        "Gemini 2.5 Flash": "gemini-2.5-flash",
        "Gemini 3.5 Flash": "gemini-3.5-flash"
    }

    nume_model_selectat = st.selectbox("Alege modelul Google:", list(modele_disponibile.keys()))
    model_id = modele_disponibile[nume_model_selectat]
    
    if not api_key_env:
        api_key_manual = st.text_input("Introdu Google API Key:", type="password")
    else:
        api_key_manual = api_key_env

    if st.button("🗑️ Șterge Conversația"):
        st.session_state.mesaje = []
        st.session_state.memorie.clear()
        st.rerun()

# GESTIONAREA MEMORIEI ---
if "mesaje" not in st.session_state:
    st.session_state.mesaje = []
if "memorie" not in st.session_state:
    st.session_state.memorie = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

for mesaj in st.session_state.mesaje:
    with st.chat_message(mesaj["rol"]):
        st.markdown(mesaj["continut"])

# LOGICA DE RĂSPUNS A BOTULUI ---
intrebare = st.chat_input("Adresează o întrebare medicală...")

if intrebare:
    if not api_key_manual:
        st.error("Te rog să configurezi Google API Key!")
        st.stop()

    with st.chat_message("user"):
        st.markdown(intrebare)
    st.session_state.mesaje.append({"rol": "user", "continut": intrebare})

    llm = ChatGoogleGenerativeAI(
        google_api_key=api_key_manual, 
        model=model_id, 
        temperature=0.2 
    )

    # Lanțul clasic, construit cu motor de căutare MMR extins
    lant_conversatie = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(
            search_type="mmr", 
            search_kwargs={
                "k": 25,         # fragmente
                "fetch_k": 80    # cele mai relevante
            } 
        ),
        memory=st.session_state.memorie,
        combine_docs_chain_kwargs={"prompt": PROMPT_MEDICAL},
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        verbose=True
    )

    with st.chat_message("assistant"):
        with st.spinner(f"{nume_model_selectat} formulează răspunsul..."):
            try:
                # --- STRATUL DE NORMALIZARE DINAMICĂ (Query Pre-processing) ---
                prompt_normalizare = f"Corectează abrevierile medicale și pune majusculă la numele de medicamente din acest text. Returnează STRICT textul corectat, fără nicio altă explicație: '{intrebare}'"
                
                mesaj_normalizat = llm.invoke(prompt_normalizare)
                
                # EXTRAGERE SIGURĂ (pentru a evita eroarea cu 'list' vs 'string')
                continut = mesaj_normalizat.content
                if isinstance(continut, list):
                    # Dacă e o listă multimodală, extragem valoarea din cheia "text" a primului element
                    intrebare_optimizata = continut[0].get("text", "") if isinstance(continut[0], dict) else str(continut[0])
                else:
                    # Dacă e un string clasic
                    intrebare_optimizata = str(continut)
                
                intrebare_optimizata = intrebare_optimizata.strip()
                
                # Fallback de siguranță: dacă AI-ul nu a returnat nimic, folosim întrebarea originală
                if not intrebare_optimizata:
                    intrebare_optimizata = intrebare
                
                # Invocăm lanțul RAG cu întrebarea curățată
                raspuns_bot = lant_conversatie.invoke({"question": intrebare_optimizata})
                text_raspuns = raspuns_bot["answer"]
                st.markdown(text_raspuns)
                st.session_state.mesaje.append({"rol": "assistant", "continut": text_raspuns})
                
            except Exception as e:
                st.error(f"Eroare API: {str(e)}")