import streamlit as st
import torch
import re
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "Karyl-Maxine/disaster-distilroberta"
THRESHOLD = 0.65
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hugging Face API (from Streamlit Cloud secrets)
HF_TOKEN = st.secrets["HF_TOKEN"]

LLM_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# =====================================================
# LOAD CLASSIFIER MODEL
# =====================================================

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

def predict_emergency(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        positive_prob = probs[:, 1].item()

    return positive_prob > THRESHOLD, positive_prob

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

# =====================================================
# DISASTER CATEGORIES (Expanded)
# =====================================================

EMERGENCY_KEYWORDS = {
    "fire": ["fire", "wildfire", "burning", "flames"],
    "flood": ["flood", "flooded", "flash flood"],
    "earthquake": ["earthquake", "tremor", "aftershock"],
    "storm": ["storm", "typhoon", "hurricane", "cyclone"],
    "landslide": ["landslide", "mudslide"],
    "tsunami": ["tsunami"],
    "explosion": ["explosion", "blast", "bomb"],
    "collapse": ["collapse", "collapsed", "building collapse"],
    "accident": ["accident", "crash", "collision"],
    "injury": ["injured", "bleeding", "unconscious", "casualties"]
}

def detect_incidents(text):
    detected = []
    for category, words in EMERGENCY_KEYWORDS.items():
        if any(word in text for word in words):
            detected.append(category)
    return detected

# =====================================================
# SEVERITY + DISPATCH
# =====================================================

def calculate_severity(confidence, categories):
    score = confidence * 2
    score += len(categories)

    if "injury" in categories:
        score += 2

    if score >= 5:
        return "Critical"
    elif score >= 4:
        return "High"
    elif score >= 3:
        return "Moderate"
    else:
        return "Low"

DISPATCH_MAP = {
    "fire": "Fire Department",
    "flood": "Flood Rescue Unit",
    "earthquake": "Search and Rescue Team",
    "storm": "Disaster Response Team",
    "landslide": "Geological Response Unit",
    "tsunami": "Coastal Emergency Unit",
    "explosion": "Bomb Disposal Unit",
    "collapse": "Urban Search and Rescue",
    "accident": "Traffic Emergency Unit",
    "injury": "Medical Emergency Services"
}

def determine_dispatch(categories):
    units = set()
    for cat in categories:
        if cat in DISPATCH_MAP:
            units.add(DISPATCH_MAP[cat])
    return list(units) if units else ["Disaster Response Team"]

# =====================================================
# RAG KNOWLEDGE BASE
# =====================================================

DOCS = [
    "Evacuate immediately if there is visible fire.",
    "Stay low to avoid smoke inhalation.",
    "Move to higher ground during floods.",
    "Drop, cover, and hold during earthquakes.",
    "Avoid unstable structures after a collapse.",
    "Do not approach explosive devices.",
    "Check for injuries and apply first aid if trained.",
    "Stay away from landslide-prone slopes.",
    "Avoid driving through flooded roads.",
    "Monitor official disaster announcements."
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def build_index(docs):
    embeddings = embedder.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

index = build_index(DOCS)

def rag_retrieve(query, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [DOCS[i] for i in indices[0]]

# =====================================================
# LLM GENERATION (HF API)
# =====================================================

def query_llm(prompt):

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.6,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(
            LLM_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            return "⚠️ LLM is currently loading or rate-limited. Please try again."

        return response.json()[0]["generated_text"]

    except requests.exceptions.RequestException:
        return "⚠️ Unable to connect to LLM service."

# =====================================================
# AGENTIC PIPELINE
# =====================================================

def process_message(user_input):

    state = st.session_state.incident_state

    cleaned = clean_text(user_input)
    is_emergency, confidence = predict_emergency(cleaned)

    if not is_emergency:
        return "No emergency detected. Please provide more details if this is urgent."

    detected = detect_incidents(cleaned)

    state["confidence"] = max(state["confidence"], confidence)
    state["categories"] = list(set(state["categories"] + detected))

    severity = calculate_severity(state["confidence"], state["categories"])
    dispatch = determine_dispatch(state["categories"])

    rag_docs = rag_retrieve(cleaned)
    context = "\n".join(rag_docs)

    prompt = f"""
You are an intelligent emergency response AI.

Incident Description:
{user_input}

Detected Disaster Types:
{', '.join(state['categories'])}

Severity Level:
{severity}

Relevant Safety Guidelines:
{context}

Generate:
1. Step-by-step emergency instructions.
2. Immediate life-saving actions.
3. Safety precautions.
4. A short reassuring message.
"""

    generated_advice = query_llm(prompt)

    return f"""
🚨 Emergency Confirmed

Incident Types: {', '.join(state['categories'])}
Severity Level: {severity}
Dispatch Units: {', '.join(dispatch)}
Confidence: {round(state['confidence'],3)}

{generated_advice}
"""

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(page_title="Agentic RespondrAI", page_icon="🚨")
st.title("🚨 Agentic Conversational Emergency Response AI")

if "incident_state" not in st.session_state:
    st.session_state.incident_state = {
        "confidence": 0.0,
        "categories": []
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Describe the situation:")

if st.button("Send"):
    if user_input:
        st.session_state.chat_history.append(("User", user_input))
        response = process_message(user_input)
        st.session_state.chat_history.append(("Agent", response))

for speaker, message in st.session_state.chat_history:
    if speaker == "User":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Agent:** {message}")

if st.session_state.incident_state["confidence"] > 0:
    st.write("### Current Confidence Level")
    st.progress(st.session_state.incident_state["confidence"])