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
# LOAD BINARY CLASSIFIER (UNCHANGED)
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
# SEMANTIC INCIDENT CLASSIFIER
# =============================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

DISASTER_CATEGORIES = [
    "fire",
    "flood",
    "earthquake",
    "storm",
    "tornado",
    "tsunami",
    "landslide",
    "volcano",
    "explosion",
    "collapse",
    "drought",
    "pandemic",
    "shooting",
    "chemical spill"
]

category_embeddings = embedder.encode(
    DISASTER_CATEGORIES,
    normalize_embeddings=True
)

def detect_incidents_semantic(text, threshold=0.40, top_k=2):
    query_embedding = embedder.encode(
        [text],
        normalize_embeddings=True
    )

    similarities = np.dot(category_embeddings, query_embedding.T).squeeze()
    top_indices = similarities.argsort()[-top_k:][::-1]

    detected = []
    for idx in top_indices:
        if similarities[idx] >= threshold:
            detected.append(DISASTER_CATEGORIES[idx])

    return detected

# =============================
# STRONG CATEGORY-BASED RAG
# =============================

KNOWLEDGE_BASE = {
    "fire": [
        "Evacuate immediately and call the fire department.",
        "Do not use elevators during a fire.",
        "Stay low to avoid smoke inhalation.",
        "Use fire extinguishers only if trained.",
        "Check doors for heat before opening."
    ],
    "flood": [
        "Move to higher ground immediately.",
        "Avoid walking or driving through floodwaters.",
        "Turn off electricity if safe to do so.",
        "Prepare emergency supplies and clean water.",
        "Follow local evacuation advisories."
    ],
    "earthquake": [
        "Drop, cover, and hold on.",
        "Stay away from windows and heavy objects.",
        "After shaking stops, check for injuries.",
        "Do not reenter damaged buildings.",
        "Expect aftershocks."
    ],
    "storm": [
        "Stay indoors and away from windows.",
        "Secure outdoor objects.",
        "Monitor official weather updates.",
        "Prepare emergency kit.",
        "Avoid flooded roads."
    ],
    "explosion": [
        "Move away from the blast area immediately.",
        "Check for secondary hazards.",
        "Assist injured if safe.",
        "Avoid unstable structures.",
        "Contact emergency responders."
    ],
    "collapse": [
        "Evacuate unstable structures immediately.",
        "Avoid entering damaged buildings.",
        "Call search and rescue teams.",
        "Watch for falling debris.",
        "Check for trapped individuals."
    ],
    "pandemic": [
        "Follow public health guidelines.",
        "Wear protective masks if required.",
        "Practice social distancing.",
        "Wash hands frequently.",
        "Seek medical advice if symptomatic."
    ]
}

# Build FAISS index per category using cosine similarity
faiss_indices = {}

for category, docs in KNOWLEDGE_BASE.items():
    embeddings = embedder.encode(
        docs,
        normalize_embeddings=True
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine (normalized)
    index.add(np.array(embeddings))
    faiss_indices[category] = (index, docs)

def rag_retrieve(query, categories, k=3):
    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True
    )

    results = []

    for category in categories:
        if category in faiss_indices:
            index, docs = faiss_indices[category]
            distances, indices = index.search(
                np.array(query_embedding),
                min(k, len(docs))
            )
            for i in indices[0]:
                results.append(docs[i])

    return list(set(results))  # remove duplicates

# =============================
# SEVERITY SCORING
# =============================

def calculate_severity(confidence, detected_categories):

    score = 0
    score += confidence * 2
    score += len(detected_categories)

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
    "storm": "Disaster Response Unit",
    "explosion": "Bomb Disposal Unit",
    "collapse": "Urban Search and Rescue",
    "pandemic": "Public Health Emergency Unit"
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

    detected_categories = detect_incidents_semantic(cleaned)
    rag_docs = rag_retrieve(cleaned, detected_categories)

    severity = calculate_severity(confidence, detected_categories)
    dispatch_units = determine_dispatch(detected_categories)

    return {
        "emergency": True,
        "types": detected_categories if detected_categories else ["unknown"],
        "severity": severity,
        "confidence": round(confidence, 3),
        "dispatch": dispatch_units,
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

            st.write("### Recommended Actions")
            for a in result["actions"]:
                st.write("-", a)