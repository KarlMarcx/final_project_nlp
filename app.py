import streamlit as st
import torch
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import openai

# ==========================
# CONFIG
# ==========================
CLASSIFIER_MODEL = "Karyl-Maxine/disaster-distilroberta"
THRESHOLD = 0.65
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Groq API setup (OpenAI compatible free-tier)
groq_api_key = st.secrets["GROQ"]["API_KEY"]
groq_base_url = "https://api.groq.com/openai/v1"
client = openai.OpenAI(api_key=groq_api_key, base_url=groq_base_url)

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_classifier():
    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)
    model.to(device)
    model.eval()
    return model, tokenizer

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

classifier_model, classifier_tokenizer = load_classifier()
embedder = load_embedder()

# ==========================
# UTILITY FUNCTIONS
# ==========================
def predict_emergency(text):
    inputs = classifier_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = classifier_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        score = probs[:, 1].item()

    return score > THRESHOLD, score

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

# ==========================
# CATEGORY DETECTION
# ==========================
EMERGENCY_CATEGORIES = [
    "fire", "flood", "earthquake", "storm",
    "shooting", "violence", "medical emergency",
    "death incident", "accident"
]
category_embeddings = embedder.encode(EMERGENCY_CATEGORIES, normalize_embeddings=True)

def detect_categories(text, threshold=0.25, top_k=3):
    query_embedding = embedder.encode([text], normalize_embeddings=True)
    similarities = np.dot(category_embeddings, query_embedding.T).squeeze()
    top_indices = similarities.argsort()[-top_k:][::-1]

    detected = []
    for idx in top_indices:
        if similarities[idx] >= threshold:
            detected.append(EMERGENCY_CATEGORIES[idx])

    keyword_map = {
        "fire": "fire",
        "burn": "fire",
        "violence": "violence",
        "crash": "accident",
        "injured": "medical emergency",
        "dead": "death incident",
        "shoot": "shooting",
    }
    for key, value in keyword_map.items():
        if key in text:
            detected.append(value)

    return list(set(detected))

# ==========================
# KNOWLEDGE BASE + FAISS
# ==========================
KNOWLEDGE_BASE = {
    "fire": ["Evacuate immediately.", "Do not use elevators.", "Stay low to avoid smoke."],
    "violence": ["Move to a secure location.", "Avoid confrontation.", "Contact law enforcement immediately."],
    "accident": ["Ensure scene safety.", "Call emergency responders.", "Do not move severely injured individuals."],
    "medical emergency": ["Call medical services immediately.", "Check breathing and pulse.", "Provide first aid if trained."],
}
GENERAL_DOCS = ["Ensure your own safety first.", "Call emergency services immediately.", "Assist others only if safe."]

@st.cache_resource
def build_faiss():
    faiss_indices = {}
    for category, docs in KNOWLEDGE_BASE.items():
        embeddings = embedder.encode(docs, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.array(embeddings))
        faiss_indices[category] = (index, docs)

    general_embeddings = embedder.encode(GENERAL_DOCS, normalize_embeddings=True)
    general_index = faiss.IndexFlatIP(general_embeddings.shape[1])
    general_index.add(np.array(general_embeddings))
    return faiss_indices, general_index

faiss_indices, general_index = build_faiss()

def rag_retrieve(query, categories, k=2):
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    results = []

    for category in categories:
        if category in faiss_indices:
            index, docs = faiss_indices[category]
            _, indices = index.search(np.array(query_embedding), min(k, len(docs)))
            for i in indices[0]:
                results.append(docs[i])

    if not results:
        _, indices = general_index.search(np.array(query_embedding), min(k, len(GENERAL_DOCS)))
        results = [GENERAL_DOCS[i] for i in indices[0]]

    return list(set(results))

# ==========================
# SEVERITY & DISPATCH
# ==========================
def calculate_severity(confidence, categories):
    score = confidence * 2 + len(categories)
    if "death incident" in categories: score += 3
    if "violence" in categories: score += 2
    if "fire" in categories: score += 2
    if "medical emergency" in categories: score += 1.5

    if score >= 6: return "Critical"
    elif score >= 4: return "High"
    elif score >= 3: return "Moderate"
    else: return "Low"

def generate_dispatch_units(categories, severity):
    units = set()
    category_map = {
        "fire": ["Fire Department", "Medical Emergency Services"],
        "violence": ["Police Department", "Rapid Response Unit"],
        "accident": ["Traffic/Rescue Unit", "Medical Emergency Services"],
        "medical emergency": ["Ambulance Services", "First Aid Team"],
        "death incident": ["Medical Examiner", "Police Department"],
        "shooting": ["Police Department", "SWAT Team"],
    }
    severity_map = {
        "Low": ["Community Monitoring Unit"],
        "Moderate": ["Local Emergency Response Team"],
        "High": ["City Emergency Task Force"],
        "Critical": ["National Disaster Response Force", "Crisis Management Authority"],
    }

    for cat in categories:
        if cat in category_map:
            units.update(category_map[cat])
    if severity in severity_map:
        units.update(severity_map[severity])
    return list(units)

# ==========================
# SUMMARY WITH GROQ FREE-TIER
# ==========================
def generate_summary_groq(text):
    prompt = f"Summarize the following incident in 2–3 sentences:\n\n{text}"
    try:
        response = client.chat.completions.create(
            model="nous-hermes-13b-mini",   # free-tier model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Groq API error: {e}")
        return "Summary generation failed."

# ==========================
# PIPELINE
# ==========================
def respondrAI_pipeline(text):
    cleaned = clean_text(text)
    is_emergency, confidence = predict_emergency(cleaned)
    if not is_emergency:
        return {"emergency": False, "confidence": round(confidence,3), "message": "No emergency detected."}

    categories = detect_categories(cleaned)
    docs = rag_retrieve(cleaned, categories)
    severity = calculate_severity(confidence, categories)
    dispatch = generate_dispatch_units(categories, severity)

    # Spinner for user feedback
    with st.spinner("Generating AI summary…"):
        summary = generate_summary_groq(text)

    actions = " ".join(docs)
    authorities = "Local police, fire department, and medical emergency services."
    advice = "Follow official instructions and prioritize personal safety."

    report = f"""
Situation Summary:
{summary}

Risk Level:
The situation is assessed as {severity} based on detected incident categories.

Immediate Actions:
{actions}

Recommended Authorities:
{authorities}

Safety Advice:
{advice}
"""
    return {
        "emergency": True,
        "types": categories if categories else ["general emergency"],
        "severity": severity,
        "confidence": round(confidence,3),
        "retrieved": docs,
        "dispatch": dispatch,
        "report": report
    }

# ==========================
# STREAMLIT UI
# ==========================
st.set_page_config(page_title="RespondrAI Generative RAG", page_icon="🚨")
st.title("🚨 RespondrAI - Emergency Agent")

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
            st.write("### Severity")
            st.write(result["severity"])
            st.write("### Confidence")
            st.progress(result["confidence"])

            st.write("### Retrieved Knowledge")
            for doc in result["retrieved"]:
                st.write("-", doc)

            st.write("### Dispatch Units")
            for unit in result["dispatch"]:
                st.write("-", unit)

            st.write("### 🤖 AI Generated Report")
            st.write(result["report"])