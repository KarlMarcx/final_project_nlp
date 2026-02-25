# import streamlit as st
# import torch
# import re
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # =============================
# # CONFIGURATION
# # =============================

# MODEL_PATH = "Karyl-Maxine/disaster-distilroberta"
# THRESHOLD = 0.65
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # =============================
# # LOAD BINARY CLASSIFIER (UNCHANGED)
# # =============================

# @st.cache_resource
# def load_model():
#     model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#     model.to(device)
#     model.eval()
#     return model, tokenizer

# model, tokenizer = load_model()

# def predict_emergency(text):
#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=128
#     ).to(device)

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs.logits, dim=1)
#         positive_prob = probs[:, 1].item()

#     return positive_prob > THRESHOLD, positive_prob

# # =============================
# # TEXT CLEANING
# # =============================

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"http\S+", "", text)
#     text = re.sub(r"[^a-zA-Z\s]", "", text)
#     return text.strip()

# # =============================
# # SEMANTIC CATEGORY DETECTION
# # =============================

# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# EMERGENCY_CATEGORIES = [
#     "fire",
#     "flood",
#     "earthquake",
#     "storm",
#     "tornado",
#     "tsunami",
#     "landslide",
#     "volcano",
#     "explosion",
#     "collapse",
#     "drought",
#     "pandemic",
#     "shooting",
#     "violence",
#     "medical emergency",
#     "death incident",
#     "accident",
#     "chemical spill",
#     "public disturbance"
# ]

# category_embeddings = embedder.encode(
#     EMERGENCY_CATEGORIES,
#     normalize_embeddings=True
# )

# def detect_incidents_semantic(text, threshold=0.38, top_k=2):
#     query_embedding = embedder.encode(
#         [text],
#         normalize_embeddings=True
#     )

#     similarities = np.dot(category_embeddings, query_embedding.T).squeeze()
#     top_indices = similarities.argsort()[-top_k:][::-1]

#     detected = []
#     for idx in top_indices:
#         if similarities[idx] >= threshold:
#             detected.append(EMERGENCY_CATEGORIES[idx])

#     return detected

# # =============================
# # KNOWLEDGE BASE (EXPANDED)
# # =============================

# KNOWLEDGE_BASE = {

#     "fire": [
#         "Evacuate immediately and contact the fire department.",
#         "Do not use elevators during a fire.",
#         "Stay low to avoid smoke inhalation.",
#         "Use fire extinguishers only if trained."
#     ],

#     "flood": [
#         "Move to higher ground immediately.",
#         "Avoid walking or driving through floodwaters.",
#         "Turn off electricity if safe.",
#         "Follow evacuation advisories."
#     ],

#     "earthquake": [
#         "Drop, cover, and hold on.",
#         "Stay away from windows.",
#         "Check for injuries after shaking stops.",
#         "Avoid reentering damaged buildings."
#     ],

#     "medical emergency": [
#         "Call emergency medical services immediately.",
#         "Check for breathing and pulse.",
#         "Begin CPR if trained.",
#         "Keep the person stable until responders arrive."
#     ],

#     "death incident": [
#         "Contact emergency authorities immediately.",
#         "Do not disturb the scene.",
#         "Allow medical professionals to assess.",
#         "Provide information to responders."
#     ],

#     "shooting": [
#         "Seek immediate shelter.",
#         "Call law enforcement.",
#         "Avoid confrontation.",
#         "Provide first aid if safe."
#     ],

#     "accident": [
#         "Ensure scene safety.",
#         "Call emergency responders.",
#         "Assist injured if safe.",
#         "Do not move seriously injured victims."
#     ],

#     "violence": [
#         "Move to a safe location.",
#         "Contact authorities immediately.",
#         "Avoid escalating the situation."
#     ]
# }

# GENERAL_EMERGENCY_DOCS = [
#     "Call emergency services immediately.",
#     "Ensure your own safety first.",
#     "Provide clear information to responders.",
#     "Assist others only if it is safe to do so."
# ]

# # =============================
# # BUILD FAISS INDICES (COSINE)
# # =============================

# faiss_indices = {}

# for category, docs in KNOWLEDGE_BASE.items():
#     embeddings = embedder.encode(
#         docs,
#         normalize_embeddings=True
#     )
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(np.array(embeddings))
#     faiss_indices[category] = (index, docs)

# general_embeddings = embedder.encode(
#     GENERAL_EMERGENCY_DOCS,
#     normalize_embeddings=True
# )

# dim = general_embeddings.shape[1]
# general_index = faiss.IndexFlatIP(dim)
# general_index.add(np.array(general_embeddings))

# def rag_retrieve(query, categories, k=3):
#     query_embedding = embedder.encode(
#         [query],
#         normalize_embeddings=True
#     )

#     results = []

#     # If no specific category detected → fallback
#     if not categories:
#         distances, indices = general_index.search(
#             np.array(query_embedding),
#             min(k, len(GENERAL_EMERGENCY_DOCS))
#         )
#         return [GENERAL_EMERGENCY_DOCS[i] for i in indices[0]]

#     for category in categories:
#         if category in faiss_indices:
#             index, docs = faiss_indices[category]
#             distances, indices = index.search(
#                 np.array(query_embedding),
#                 min(k, len(docs))
#             )
#             for i in indices[0]:
#                 results.append(docs[i])

#     if not results:
#         # Safety fallback
#         distances, indices = general_index.search(
#             np.array(query_embedding),
#             min(k, len(GENERAL_EMERGENCY_DOCS))
#         )
#         return [GENERAL_EMERGENCY_DOCS[i] for i in indices[0]]

#     return list(set(results))

# # =============================
# # SEVERITY
# # =============================

# def calculate_severity(confidence, detected_categories):
#     score = confidence * 2 + len(detected_categories)

#     if "death incident" in detected_categories:
#         score += 2
#     if "medical emergency" in detected_categories:
#         score += 1.5

#     if score >= 5:
#         return "Critical"
#     elif score >= 4:
#         return "High"
#     elif score >= 3:
#         return "Moderate"
#     else:
#         return "Low"

# # =============================
# # DISPATCH
# # =============================

# DISPATCH_MAP = {
#     "fire": "Fire Department",
#     "flood": "Flood Rescue Unit",
#     "earthquake": "Search and Rescue Team",
#     "medical emergency": "Medical Emergency Services",
#     "death incident": "Emergency Medical Services",
#     "shooting": "Police Department",
#     "violence": "Law Enforcement",
#     "accident": "Emergency Response Unit"
# }

# def determine_dispatch(categories):
#     units = set()
#     for cat in categories:
#         if cat in DISPATCH_MAP:
#             units.add(DISPATCH_MAP[cat])
#     return list(units) if units else ["Disaster Response Team"]

# # =============================
# # MAIN PIPELINE
# # =============================

# def respondrAI_pipeline(text):

#     cleaned = clean_text(text)
#     is_emergency, confidence = predict_emergency(cleaned)

#     if not is_emergency:
#         return {
#             "emergency": False,
#             "confidence": round(confidence, 3),
#             "message": "No emergency detected."
#         }

#     detected_categories = detect_incidents_semantic(cleaned)
#     rag_docs = rag_retrieve(cleaned, detected_categories)

#     severity = calculate_severity(confidence, detected_categories)
#     dispatch_units = determine_dispatch(detected_categories)

#     return {
#         "emergency": True,
#         "types": detected_categories if detected_categories else ["general emergency"],
#         "severity": severity,
#         "confidence": round(confidence, 3),
#         "dispatch": dispatch_units,
#         "actions": rag_docs
#     }

# # =============================
# # STREAMLIT UI
# # =============================

# st.set_page_config(page_title="RespondrAI Hybrid Agent", page_icon="🚨")
# st.title("🚨 RespondrAI - Advanced Hybrid Emergency Agent")

# user_input = st.text_area("Enter an incident report or tweet:")

# if st.button("Analyze"):
#     if not user_input.strip():
#         st.warning("Please enter text.")
#     else:
#         result = respondrAI_pipeline(user_input)

#         if not result["emergency"]:
#             st.success(result["message"])
#             st.write("Confidence:", result["confidence"])
#         else:
#             st.error("🚨 Emergency Detected!")

#             st.write("### Incident Types")
#             st.write(", ".join(result["types"]))

#             st.write("### Severity Level")
#             st.write(result["severity"])

#             st.write("### Dispatch Units")
#             for unit in result["dispatch"]:
#                 st.write("-", unit)

#             st.write("### Confidence Score")
#             st.progress(result["confidence"])
#             st.write(f"{result['confidence'] * 100:.1f}%")

#             st.write("### Recommended Actions")
#             for a in result["actions"]:
#                 st.write("-", a)

import streamlit as st
import torch
import re
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

# ===========================
# CONFIG
# ===========================
CLASSIFIER_MODEL = "Karyl-Maxine/disaster-distilroberta"
GEN_MODEL = "google/flan-t5-base"
THRESHOLD = 0.65
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# LOAD MODELS
# ===========================
@st.cache_resource
def load_classifier():
    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)
    model.to(device)
    model.eval()
    return model, tokenizer

classifier_model, classifier_tokenizer = load_classifier()


def predict_emergency(text):
    inputs = classifier_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = classifier_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        score = probs[:, 1].item()
    return score > THRESHOLD, score


@st.cache_resource
def load_generator():
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model.to(device)
    model.eval()
    return model, tokenizer

gen_model, gen_tokenizer = load_generator()


def generate_response(prompt):
    inputs = gen_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,        # <- turn sampling back on
            temperature=0.3,       # <- low randomness
            top_p=0.9,
            repetition_penalty=1.2
        )

    decoded = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()


# ===========================
# CLEAN TEXT
# ===========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()


# ===========================
# CATEGORY DETECTION
# ===========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

EMERGENCY_CATEGORIES = [
    "fire", "flood", "earthquake", "storm",
    "shooting", "violence", "medical emergency",
    "death incident", "accident"
]

category_embeddings = embedder.encode(EMERGENCY_CATEGORIES, normalize_embeddings=True)


def detect_categories(text, threshold=0.30, top_k=3):
    query_embedding = embedder.encode([text], normalize_embeddings=True)
    similarities = np.dot(category_embeddings, query_embedding.T).squeeze()
    top_indices = similarities.argsort()[-top_k:][::-1]

    detected = []
    for idx in top_indices:
        if similarities[idx] >= threshold:
            detected.append(EMERGENCY_CATEGORIES[idx])

    # Keyword safety net
    keyword_map = {
        "fire": "fire",
        "burn": "fire",
        "ablaze": "fire",
        "violence": "violence",
        "riot": "violence",
        "stone": "violence",
        "crash": "accident",
        "injured": "medical emergency",
        "dead": "death incident",
        "shoot": "shooting"
    }

    text_lower = text.lower()
    for key, value in keyword_map.items():
        if key in text_lower:
            detected.append(value)

    return list(set(detected))


# ===========================
# KNOWLEDGE BASE
# ===========================
KNOWLEDGE_BASE = {
    "fire": ["Evacuate immediately.", "Do not use elevators.", "Stay low to avoid smoke."],
    "violence": ["Move to a secure location.", "Avoid confrontation.", "Contact law enforcement immediately."],
    "accident": ["Ensure scene safety.", "Call emergency responders.", "Do not move severely injured individuals."],
    "medical emergency": ["Call medical services immediately.", "Check breathing and pulse.", "Provide first aid if trained."]
}

GENERAL_DOCS = ["Ensure your own safety first.", "Call emergency services immediately.", "Assist others only if safe."]

# ===========================
# BUILD FAISS INDEX
# ===========================
faiss_indices = {}
for category, docs in KNOWLEDGE_BASE.items():
    embeddings = embedder.encode(docs, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))
    faiss_indices[category] = (index, docs)

general_embeddings = embedder.encode(GENERAL_DOCS, normalize_embeddings=True)
general_index = faiss.IndexFlatIP(general_embeddings.shape[1])
general_index.add(np.array(general_embeddings))


def rag_retrieve(query, categories, k=2):
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    results = []
    for category in categories:
        if category in faiss_indices:
            index, docs = faiss_indices[category]
            distances, indices = index.search(np.array(query_embedding), min(k, len(docs)))
            for i in indices[0]:
                results.append(docs[i])
    if not results:
        distances, indices = general_index.search(np.array(query_embedding), min(k, len(GENERAL_DOCS)))
        results = [GENERAL_DOCS[i] for i in indices[0]]
    return list(set(results))


# ===========================
# SEVERITY
# ===========================
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


# ===========================
# PROMPT BUILDER (NO ROLE AT TOP)
# ===========================
def build_prompt(user_text, docs, categories, severity):
    context = " ".join(docs)

    return f"""
Task: Generate a structured emergency response report.

Incident:
{user_text}

Detected Types: {', '.join(categories)}
Severity Level: {severity}

Safety Guidelines:
{context}

Output:
Situation Summary:
Risk Level:
Immediate Actions:
Recommended Authorities:
Safety Advice:
"""

# ===========================
# PIPELINE
# ===========================
def respondrAI_pipeline(text):
    cleaned = clean_text(text)
    is_emergency, confidence = predict_emergency(cleaned)
    if not is_emergency:
        return {"emergency": False, "confidence": round(confidence, 3), "message": "No emergency detected."}

    categories = detect_categories(cleaned)
    docs = rag_retrieve(cleaned, categories)
    severity = calculate_severity(confidence, categories)
    prompt = build_prompt(text, docs, categories, severity)
    report = generate_response(prompt)
    return {
        "emergency": True,
        "types": categories if categories else ["general emergency"],
        "severity": severity,
        "confidence": round(confidence, 3),
        "retrieved": docs,
        "report": report
    }


# ===========================
# STREAMLIT UI
# ===========================
st.set_page_config(page_title="RespondrAI Generative RAG", page_icon="🚨")
st.title("🚨 RespondrAI - Robust Generative RAG Emergency Agent")

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
            st.write("### 🤖 AI Generated Report")
            st.write(result["report"])