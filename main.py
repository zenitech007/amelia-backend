from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from google import genai
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import uuid

# =========================
# LOAD ENV
# =========================

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# =========================
# FASTAPI SERVER
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# JSON DATA CATCHER
# =========================
# UPDATED: Now catches the Base64 image string from the Next.js frontend!
class ChatRequest(BaseModel):
    user_message: str
    session_id: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    image_data: Optional[str] = None

# =========================
# AI BRAINS
# =========================

gemini_client = genai.Client(api_key=GEMINI_KEY)
openai_client = OpenAI(api_key=OPENAI_KEY)

# =========================
# CONVERSATION MEMORY
# =========================

memory_store = {}

def get_memory(session_id):
    if session_id not in memory_store:
        memory_store[session_id] = []
    return memory_store[session_id]

# =========================
# MEDICAL KNOWLEDGE BASE
# =========================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="medical_kb")

def load_documents():
    path = "medical_kb/documents.txt"
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]

    if collection.count() == 0 and docs:
        for i, doc in enumerate(docs):
            embedding = embedding_model.encode(doc).tolist()
            collection.add(
                ids=[str(i)],
                documents=[doc],
                embeddings=[embedding]
            )

load_documents()

def retrieve_medical_context(query):
    doc_count = collection.count()
    if doc_count == 0:
        return "No additional medical context available."

    embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(3, doc_count) 
    )

    if not results["documents"] or not results["documents"][0]:
         return "No additional medical context available."

    docs = results["documents"][0]
    return "\n".join(docs)

# =========================
# EMERGENCY DETECTION
# =========================

EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "can't breathe", "shortness of breath",
    "stroke", "face drooping", "arm weakness", "slurred speech",
    "unconscious", "severe bleeding"
]

def detect_emergency(message):
    msg = message.lower()
    for word in EMERGENCY_KEYWORDS:
        if word in msg:
            return True
    return False

# =========================
# SYMPTOM TRIAGE
# =========================

def triage_level(message):
    triage_prompt = f"""
    Classify the medical urgency of this message.
    Levels: EMERGENCY, URGENT, NON-URGENT
    Message: {message}
    Reply only with the level.
    """
    response = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=triage_prompt
    )
    try:
        return (response.text or "NON-URGENT").strip()
    except:
        return "NON-URGENT"

# =========================
# SYSTEM PROMPT
# =========================

system_prompt = """
You are A.M.E.L.I.A. (Advanced Medical Expert Learning & Intelligence Agent).
You are a warm, highly empathetic, and incredibly smart personal healthcare assistant.

CRITICAL INSTRUCTIONS: 
1. PROFILE AWARENESS: You MUST actively cross-reference your advice with the patient's Age, Weight, Allergies, Conditions, and Medications.
2. VISION AWARENESS (LABS & DIET): 
   - If the user uploads an image of food, analyze the estimated carbohydrates, sugars, and calories, and advise them on how it impacts their specific conditions (e.g., blood sugar levels).
   - If the user uploads an image of a medical report or lab result, extract the data and translate the medical jargon into simple, plain language. Point out any abnormal levels.

Tone & Personality:
- Friendly, conversational, and deeply caring. 
- Always use their first name naturally in the conversation.
- AVOID sounding like a robotic medical textbook.

Safety Rules:
- You cannot officially diagnose a disease or prescribe exact medication dosages. 
- Always gently remind them to verify critical decisions with their physical doctor.
"""

# =========================
# CHAT ENDPOINT
# =========================

@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.user_message
    session_id = request.session_id or str(uuid.uuid4())
    profile = request.profile or {}
    image_data = request.image_data  # <-- SECURING THE IMAGE DATA

    memory = get_memory(session_id)

    # ---> FORMATTING THE PATIENT PROFILE FOR THE AI <---
    first_name = profile.get("firstName", "Patient")
    age = profile.get("age", "Unknown")
    gender = profile.get("gender", "Unknown")
    weight = profile.get("weightKg", "Unknown")
    blood_type = profile.get("bloodType", "Unknown")
    genotype = profile.get("genotype", "Unknown")
    body_shape = profile.get("bodyShape", "Unknown")
    sugar_level = profile.get("sugarLevel", "Unknown")
    is_pregnant = profile.get("isPregnant", False)
    preg_status = "Yes" if is_pregnant else "No"

    conditions = profile.get("conditions", "None")
    allergies = profile.get("allergies", "None")
    meds = profile.get("currentMeds", "None")
    language = profile.get("language", "English")

    patient_context = f"""
    --- PATIENT BIOMETRICS & HISTORY ---
    Name: {first_name}
    Age: {age} | Gender: {gender} | Weight: {weight}kg
    Blood Type: {blood_type} | Genotype: {genotype}
    Body Shape: {body_shape} | Sugar Level: {sugar_level}
    Currently Pregnant: {preg_status}
    Existing Conditions: {conditions}
    Allergies: {allergies}
    Current Medications: {meds}
    Preferred Language: {language}
    ------------------------------------
    
    CRITICAL LANGUAGE INSTRUCTION:
    You MUST reply entirely and exclusively in {language}. 
    """

    def generate_response():
        if detect_emergency(user_message):
            yield f"{first_name}, your symptoms may indicate a medical emergency. Please dial 112 or go to the nearest hospital immediately."
            return

        # If there is an image attached, we AUTOMATICALLY force the MEDICAL route
        if image_data:
            decision = "MEDICAL"
        else:
            classify_prompt = f"Is this message MEDICAL or CASUAL?\nMessage: {user_message}\nReply with only MEDICAL or CASUAL."
            classification = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=classify_prompt
            )
            try:
                decision = (classification.text or "MEDICAL").strip().upper()
            except:
                decision = "MEDICAL"

        # CASUAL ROUTE
        if "CASUAL" in decision:
            casual_prompt = f"""
            {system_prompt}
            
            {patient_context}

            The user is making casual conversation. Be friendly and use their name.

            User:
            {user_message}
            """
            response_stream = gemini_client.models.generate_content_stream(
                model='gemini-2.5-flash',
                contents=casual_prompt
            )
            
            full_answer = ""
            for chunk in response_stream:
                text_chunk = chunk.text or ""
                if text_chunk:
                    full_answer += text_chunk
                    yield text_chunk 

            memory.append({"role": "user", "content": user_message})
            memory.append({"role": "assistant", "content": full_answer})
            return

        # MEDICAL / MULTIMODAL ROUTE
        urgency = triage_level(user_message)
        context = retrieve_medical_context(user_message)

        messages: list[Any] = [{"role": "system", "content": system_prompt}]
        for m in memory[-10:]:
            messages.append(m)

        user_prompt_text = f"""
        {patient_context}

        Medical Context (RAG):
        {context}

        Urgency level: {urgency}

        Question:
        {user_message}
        """

        # --- MULTIMODAL INJECTION ---
        # If the user attached an image, we change the content from a simple String into a Vision Array!
        if image_data:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt_text},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": user_prompt_text
            })

        response_stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True 
        )

        full_answer = ""
        for chunk in response_stream:
            text_chunk = chunk.choices[0].delta.content or ""
            if text_chunk:
                full_answer += text_chunk
                yield text_chunk 

        memory.append({"role": "user", "content": user_message})
        memory.append({"role": "assistant", "content": full_answer})

    return StreamingResponse(generate_response(), media_type="text/plain")

@app.get("/")
def home():
    return {"status": "AMELIA Backend is Online", "version": "1.0.0"}