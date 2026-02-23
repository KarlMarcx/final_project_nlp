import streamlit as st
import torch
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =============================
# CONFIGURATION
# =============================

MODEL_PATH = "Karyl-Maxine/disaster-distilroberta"
THRESHOLD = 0.65
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# LOAD CLASSIFICATION MODEL
# =============================

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

# =============================
# TEXT CLEANING
# =============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

# =============================
# KNOWLEDGE BASE (RAG)
# =============================

DOCS = [
    "Fire emergencies require immediate evacuation and contacting the fire department.",
    "Do not use elevators during a fire.",
    "Stay low to avoid smoke inhalation.",
    "Earthquakes require drop, cover, and hold.",
    "After an earthquake, check for structural damage before reentering buildings.",
    "Flood areas require moving to higher ground and avoiding floodwaters.",
    "Explosions may cause structural damage and secondary hazards.",
    "Call emergency services immediately if anyone is injured."
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(docs):
    embeddings = embedder.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

index = build_faiss_index(DOCS)

def rag_retrieve(query, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [DOCS[i] for i in indices[0]]

# =============================
# KEYWORD LOGIC (MULTI-DISASTER)
# =============================

EMERGENCY_KEYWORDS = {
    "fire": ["fire", "burning", "flames", "smoke", "inferno"],
    "flood": ["flood", "flooded", "rising water", "submerged"],
    "earthquake": ["earthquake", "tremor", "shake", "aftershock"],
    "explosion": ["explosion", "blast", "detonation", "boom"],
    "injury": ["injured", "bleeding", "unconscious", "wounded"]
}

def detect_incidents(text):
    detected = []
    for category, words in EMERGENCY_KEYWORDS.items():
        if any(word in text for word in words):
            detected.append(category)
    return detected

# =============================
# SEVERITY SCORING
# =============================

def calculate_severity(confidence, detected_categories):

    score = 0

    # Model confidence weight
    score += confidence * 2

    # Multiple disasters increase severity
    score += len(detected_categories)

    # Injury increases severity heavily
    if "injury" in detected_categories:
        score += 2

    if score >= 5:
        return "Critical"
    elif score >= 4:
        return "High"
    elif score >= 3:
        return "Moderate"
    else:
        return "Low"

# =============================
# DISPATCH LOGIC
# =============================

DISPATCH_MAP = {
    "fire": "Fire Department",
    "flood": "Flood Rescue Unit",
    "earthquake": "Search and Rescue Team",
    "explosion": "Bomb Disposal Unit",
    "injury": "Medical Emergency Services"
}

def determine_dispatch(categories):
    units = set()
    for cat in categories:
        if cat in DISPATCH_MAP:
            units.add(DISPATCH_MAP[cat])
    return list(units) if units else ["Disaster Response Team"]

# =============================
# MAIN PIPELINE
# =============================

def respondrAI_pipeline(text):

    cleaned = clean_text(text)
    is_emergency, confidence = predict_emergency(cleaned)

    if not is_emergency:
        return {
            "emergency": False,
            "confidence": round(confidence, 3),
            "message": "No emergency detected."
        }

    detected_categories = detect_incidents(cleaned)
    rag_docs = rag_retrieve(cleaned)

    severity = calculate_severity(confidence, detected_categories)
    dispatch_units = determine_dispatch(detected_categories)

    reasons = [
        f"Model classified this as emergency with probability {round(confidence,3)}.",
        f"Detected incident categories: {', '.join(detected_categories) if detected_categories else 'None explicitly detected.'}",
        "Retrieved relevant safety procedures from knowledge base:"
    ]

    return {
        "emergency": True,
        "types": detected_categories if detected_categories else ["unknown"],
        "severity": severity,
        "confidence": round(confidence, 3),
        "dispatch": dispatch_units,
        "reason": reasons,
        "actions": rag_docs
    }

# =============================
# STREAMLIT UI
# =============================

st.set_page_config(page_title="RespondrAI Hybrid Agent", page_icon="🚨")

st.title("🚨 RespondrAI - Advanced Hybrid Emergency Agent")

user_input = st.text_area("Enter an incident report or tweet:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter text.")
    else:
        result = respondrAI_pipeline(user_input)

        if not result["emergency"]:
            st.success(result["message"])
            st.write("Confidence:", result["confidence"])
        else:
            st.error("🚨 Emergency Detected!")

            st.write("### Incident Types")
            st.write(", ".join(result["types"]))

            st.write("### Severity Level")
            st.write(result["severity"])

            st.write("### Dispatch Units")
            for unit in result["dispatch"]:
                st.write("-", unit)

            st.write("### Confidence Score")
            st.progress(result["confidence"])
            st.write(f"{result['confidence'] * 100:.1f}%")

            st.write("### Reasoning")
            for r in result["reason"]:
                st.write("-", r)

            st.write("### Recommended Actions")
            for a in result["actions"]:
                st.write("-", a)