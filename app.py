import streamlit as st
import torch
import requests
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =============================
# HuggingFace Fine-Tuned Model
# =============================

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

    label = 1 if positive_prob > THRESHOLD else 0

    return label, positive_prob


# =============================
# Zero-Shot Incident Classifier
# =============================

HF_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

HF_HEADERS = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

def hf_zero_shot(text, labels):
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": labels}
    }

    response = requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        json=payload,
        timeout=30
    )

    if response.status_code != 200:
        raise Exception(f"HF API Error: {response.status_code}\n{response.text}")

    result = response.json()

    if isinstance(result, dict):
        result = [result]

    return result

incident_labels = [
    "fire",
    "flood",
    "earthquake",
    "hurricane",
    "explosion",
    "wildfire",
    "building collapse",
    "transport accident"
]

# =============================
# Text Cleaning
# =============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


# =============================
# Severity Agent
# =============================

class SeverityAgent:
    def assess(self, text):
        high_keywords = ["urgent", "help", "critical", "injured", "dead"]
        if any(word in text for word in high_keywords):
            return "High"
        return "Medium"

severity_agent = SeverityAgent()

# =============================
# Dispatch Agent
# =============================

class DispatchAgent:
    def route(self, incident_type):
        routing_map = {
            "fire": "Fire Department",
            "flood": "Disaster Response Team",
            "earthquake": "Disaster Response Team",
            "hurricane": "Disaster Response Team",
            "explosion": "Police & Fire Department",
            "wildfire": "Fire Department",
            "building collapse": "EMS & Fire Department",
            "transport accident": "EMS & Police"
        }
        return routing_map.get(incident_type, "General Emergency Unit")

dispatch_agent = DispatchAgent()

# =============================
# Main Pipeline
# =============================

def respondrAI_pipeline(text):

    cleaned = clean_text(text)

    # Step 1: Emergency Detection
    is_emergency, confidence = predict_emergency(cleaned)

    if is_emergency == 0:
        return {
            "emergency": False,
            "confidence": round(confidence, 3),
            "message": "No emergency detected."
        }

    # Step 2: Incident Classification
    hf_result = hf_zero_shot(cleaned, incident_labels)
    incident_type = hf_result[0]['label']

    # Step 3: Severity
    priority = severity_agent.assess(cleaned)

    # Step 4: Dispatch
    unit = dispatch_agent.route(incident_type)

    return {
        "emergency": True,
        "confidence": round(confidence, 3),
        "incident_type": incident_type,
        "priority": priority,
        "dispatch_to": unit
    }


# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="RespondrAI", page_icon="🚨")

st.title("🚨 RespondrAI")
st.markdown("Agentic AI for Emergency Incident Classification & Dispatch")

user_input = st.text_area("Enter a tweet or emergency report:")

if st.button("Analyze Incident"):

    if user_input.strip() == "":
        st.warning("Please enter text first.")
    else:
        result = respondrAI_pipeline(user_input)

        if not result["emergency"]:
            st.success("✅ No emergency detected.")
            st.write("Confidence:", result["confidence"])
        else:
            st.error("🚨 Emergency Detected!")
            st.write("Confidence:", result["confidence"])
            st.write("**Incident Type:**", result["incident_type"])
            st.write("**Priority Level:**", result["priority"])
            st.write("**Dispatch To:**", result["dispatch_to"])