import streamlit as st
import torch
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# CLASSIFICATION MODEL

MODEL_PATH = "Karyl-Maxine/disaster-distilroberta"
THRESHOLD = 0.65

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# TEXT CLEANING

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# REAL RAG SETUP

# Knowledge documents (you can expand or load from files)
DOCS = [
"Fire emergencies require immediate evacuation and contacting the fire department.",
"Do not use elevators during a fire.",
"Stay low to avoid smoke inhalation.",
"Earthquakes require drop, cover, and hold.",
"After an earthquake, check for structural damage before reentering buildings.",
"Flood areas require moving to higher ground and avoiding floodwaters.",
"Explosions may cause structural damage and secondary hazards.",
"Call emergency services if anyone is injured."
]

# Embedding model (lightweight)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed(text):
    return embedder.encode([text])

# Build FAISS vector index
doc_embeddings = np.vstack([embed(d) for d in DOCS])
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

def rag_retrieve(query, k=3):
    query_embedding = embed(query)
    distances, indices = index.search(query_embedding, k)
    return [DOCS[i] for i in indices[0]]

# KEYWORD REASONING (optional)

EMERGENCY_KEYWORDS = {
    "fire": ["fire", "massive fire", "burning", "flames", "smoke", "inferno"],
    "flood": ["flood", "flooded", "rising water", "submerged", "inundated"],
    "earthquake": ["earthquake", "tremor", "shake", "seismic", "aftershock"],
    "explosion": ["explosion", "blast", "boom", "detonation"],
    "injury": ["injured", "bleeding", "unconscious", "hurt", "wounded"]
}

def find_trigger_keywords(text):
    text = text.lower()
    matches = []

    for category, words in EMERGENCY_KEYWORDS.items():
        for word in words:
            if word in text:
                matches.append(word)

    return matches

# CLASSIFY INCIDENT TYPE (BASIC)

def classify_incident(text):
    text = text.lower()
    scores = {}

    for category, words in EMERGENCY_KEYWORDS.items():
        scores[category] = sum(word in text for word in words)

    best_match = max(scores, key=scores.get)

    if scores[best_match] > 0:
        return best_match

    return "unknown"

# PIPELINE (CLASSIFICATION + RAG)

def respondrAI_pipeline(text):

    cleaned = clean_text(text)

    is_emergency, confidence = predict_emergency(cleaned)

    if not is_emergency:
        return {
            "emergency": False,
            "confidence": round(confidence, 3),
            "message": "No emergency detected."
        }

    incident_type = classify_incident(cleaned)
    keywords = find_trigger_keywords(cleaned)

    # REAL RAG retrieval
    rag_docs = rag_retrieve(cleaned)

    # 🔥 PRO-LEVEL FALLBACK
    if incident_type == "unknown" and len(rag_docs) > 0:
        top_doc = rag_docs[0]
        inferred_type = classify_incident(top_doc)

        if inferred_type != "unknown":
            incident_type = inferred_type

    return {
        "emergency": True,
        "type": incident_type,
        "urgency": "high",
        "confidence": round(confidence, 3),
        "reason": [
            f"Input contains keywords: {keywords}" if keywords else "No direct emergency keywords detected.",
            "Knowledge base matches documents:"
        ] + rag_docs,
        "actions": rag_docs,
        "dispatch": "Fire Department" if incident_type == "fire" else "Disaster Response Team"
    }

# STREAMLIT UI

st.set_page_config(page_title="RespondrAI RAG Agent", page_icon="🚨")

st.title("🚨 RespondrAI")

user_input = st.text_area("Enter an incident report or tweet:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter text.")
    else:
        result = respondrAI_pipeline(user_input)

        if not result["emergency"]:
            st.success("No emergency detected.")
            st.write("Confidence:", result["confidence"])
        else:
            st.error("🚨 Emergency Detected!")
            st.write("Type:", result["type"])
            st.write("Urgency:", result["urgency"])
            st.write("Dispatch:", result["dispatch"])
            st.write("Confidence:", result["confidence"])

            st.write("### Reason")
            for r in result["reason"]:
                st.write(f"- {r}")

            st.write("### Recommended Actions")
            for a in result["actions"]:
                st.write(f"- {a}")