import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from google import genai
from google.genai import types
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi import BackgroundTasks
import asyncio
import os
import uuid
import base64
import json
import re
import logging
import warnings

# =========================
# SILENCE TERMINAL WARNINGS
# =========================
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# =========================
# MEDICAL RULES LOADER
# =========================

def load_json(path):
    if not os.path.exists(path):
        print(f"{path} not found.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

SYMPTOM_TRIAGE_RULES = load_json("medical_rules/symptom_triage_rules.json")
DISEASE_RULES = load_json("medical_rules/disease_probabilities.json")
DRUG_INTERACTIONS = load_json("medical_rules/drug_interactions.json")
EMERGENCY_RULES = load_json("medical_rules/emergency_conditions.json")

# =========================
# LOAD ENV
# =========================

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# =========================
# FASTAPI SERVER
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "https://web-production-691b5.up.railway.app"
        "https://amelia-tan.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# JSON DATA CATCHERS
# =========================
class ChatRequest(BaseModel):
    user_message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    image_data: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None  # NEW: Supports Dynamic Memory Sync
    medications: Optional[List[str]] = None
    age: Optional[int] = None
    weight: Optional[float] = None
    allergies: Optional[str] = None
    conditions: Optional[str] = None
    is_new_session: Optional[bool] = None

class TitleRequest(BaseModel):
    message: str

# =========================
# AI BRAINS
# =========================

# Initialize clients only if keys are present
gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# =========================
# IMAGE & BASE64 HANDLERS (NEW)
# =========================

def clean_base64(data_str: str) -> str:
    """Removes the frontend data URI prefix if it exists."""
    if "," in data_str:
        return data_str.split(",")[1]
    return data_str

def get_gemini_image_part(b64_str: str):
    """Converts a base64 string into a Gemini SDK types.Part object."""
    clean_b64 = clean_base64(b64_str)
    image_bytes = base64.b64decode(clean_b64)
    return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

# =========================
# VISION EXTRACTORS (NEW)
# =========================

def extract_lab_results(image_base64):
    """Uses Gemini Vision to parse medical lab values."""
    if not gemini_client: return ""
    prompt = """
    Extract all medical lab values from this image. Return a structured summary:
    1. Test Name | 2. Result Value | 3. Reference Range | 4. Normal/High/Low?
    """
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, get_gemini_image_part(image_base64)]
        )
        return response.text or ""
    except Exception as e:
        print(f"Lab extraction error: {e}")
        return ""

def extract_prescription_details(image_base64):
    """Uses Gemini Vision to parse drug names and schedules."""
    if not gemini_client: return ""
    prompt = """
    Extract prescription details from this image in a clear list:
    1. Medication Name | 2. Dosage | 3. Frequency | 4. Duration | 5. Special Instructions
    """
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, get_gemini_image_part(image_base64)]
        )
        return response.text or ""
    except Exception as e:
        print(f"Prescription extraction error: {e}")
        return ""

# =========================
# MEDICAL KNOWLEDGE BASE
# =========================

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="medical_kb")

def load_documents():
    path = "medical_kb/documents.txt"
    if not os.path.exists(path):
        print(f"Note: {path} not found. Starting without local medical documents.")
        return

    with open(path, "r", encoding="utf-8") as f:
        docs = [doc.strip() for doc in f.read().split("---") if doc.strip()]

    if collection.count() == 0 and docs:
        for i, doc in enumerate(docs):
            embedding = embedding_model.encode(doc).tolist()
            collection.add(
                ids=[str(i)],
                documents=[doc],
                embeddings=[embedding]
            )
        print(f"Loaded {len(docs)} documents into ChromaDB.")

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
# EMERGENCY & TRIAGE
# =========================

EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "can't breathe", "shortness of breath",
    "stroke", "face drooping", "arm weakness", "slurred speech",
    "unconscious", "severe bleeding"
]

def detect_rule_emergency(message):

    msg = message.lower()

    for emergency in EMERGENCY_RULES.get("emergencies", []):

        symptoms = emergency.get("symptoms", [])

        matches = 0

        for s in symptoms:
            if s.replace("_", " ") in msg:
                matches += 1

        if matches >= 2:
            return emergency

    return None

def triage_level(message):
    if not gemini_client:
        return "NON-URGENT"
        
    triage_prompt = f"""
    Classify the medical urgency of this message.
    Levels: EMERGENCY, URGENT, NON-URGENT
    Message: {message}
    Reply only with the level.
    """
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=triage_prompt
        )
        return (response.text or "NON-URGENT").strip()
    except Exception as e:
        print(f"Triage error: {e}")
        return "NON-URGENT"

def calculate_symptom_score(message: str):
    msg = message.lower()
    score = 0

    weights = SYMPTOM_TRIAGE_RULES.get("symptom_weights", {})

    for symptom, weight in weights.items():
        if symptom.replace("_", " ") in msg:
            score += weight

    return score

def detect_possible_diseases(message: str):
    msg = message.lower()

    results = []

    diseases = DISEASE_RULES.get("diseases", {})

    for disease, data in diseases.items():

        symptoms = data.get("symptoms", [])

        match_count = 0

        for s in symptoms:
            if s.replace("_", " ") in msg:
                match_count += 1

        if match_count >= 2:
            results.append({
                "disease": disease,
                "tests": data.get("recommended_tests", [])
            })

    return results

def detect_regional_diseases(message: str):

    msg = message.lower()

    regional_flags = []

    if "fever" in msg and "headache" in msg:
        regional_flags.append("Possible malaria symptoms")

    if "stomach pain" in msg and "fever" in msg:
        regional_flags.append("Possible typhoid symptoms")

    return regional_flags

def check_drug_interactions(medications):

    warnings = []

    for interaction in DRUG_INTERACTIONS.get("interactions", []):

        if (
            interaction["drug_a"] in medications
            and interaction["drug_b"] in medications
        ):
            warnings.append(interaction["warning"])

    return warnings

def generate_drug_safety_context(medications, allergies):

    warnings = check_drug_interactions(medications or [])

    allergy_warning = ""

    if allergies:
        allergy_warning = f"Patient has known allergies to: {allergies}. Avoid recommending related substances."

    interaction_warning = "\n".join(warnings) if warnings else "No drug interaction detected."

    return f"""
--- DRUG SAFETY CHECK ---
{interaction_warning}

{allergy_warning}
-------------------------
"""

# =========================
# DIAGNOSTIC QUESTION ENGINE
# =========================

def generate_followup_questions(message: str):

    msg = message.lower()
    questions = []

    if "headache" in msg:
        questions.append("When did the headache start?")
        questions.append("Is the pain sharp, dull, or throbbing?")

    if "fatigue" in msg or "tired" in msg:
        questions.append("How long have you been feeling unusually tired?")
        questions.append("Have you checked your blood sugar recently?")

    if "chest pain" in msg:
        questions.append("Is the pain spreading to your arm, jaw, or back?")
        questions.append("Are you experiencing shortness of breath or sweating?")

    if "fever" in msg or "headache" in msg:
        questions.append("Have you recently been exposed to mosquitoes or untreated water?")

    return questions

# =========================
# ACTIONABLE GUIDANCE
# =========================

def generate_action_steps(message: str):

    msg = message.lower()
    steps = []

    if "headache" in msg:
        steps.append("Drink a full glass of water and rest for 20 minutes.")
        steps.append("Check your blood sugar if you have a glucose monitor.")

    if "fatigue" in msg:
        steps.append("Drink at least 500ml of water within the next hour.")
        steps.append("Avoid prolonged sun exposure for the next few hours.")

    if "chest pain" in msg:
        steps.append("Stop all activity immediately.")
        steps.append("Seek emergency medical help or call emergency services.")

    return steps

# =========================
# SYSTEM PROMPT (CLINICAL AUTHORITY VERSION)
# =========================

system_prompt = """
You are **A.M.E.L.I.A** (Advanced Medical Expert Learning & Intelligence Agent), functioning as an Autonomous Clinical Intelligence System and virtual Chief Medical Officer.

--------------------------------------------------
CRITICAL OPERATIONAL COMMANDS (MANDATORY)
--------------------------------------------------
1. **NO REPETITIVE GREETINGS**: You MUST NOT start every response with "Hello", "Hi", or "Thank you for sharing". Only greet the user in the very first message of a session. For all subsequent messages, jump immediately into the medical analysis or answer.
2. **STRICT FORMATTING**: Use **Bold Text** for all section headers and key terms. You are strictly FORBIDDEN from using hashtags (#), double-bolding (****), or blockquotes for headers.
3. **MEDICATION TAGGING**: If a user confirms a dosage and frequency for a chronic medication (e.g., Metformin 500mg twice daily), you MUST output the specific system tag at the end of your response: [AMELIA_NEW_MED: Name | Dosage | Frequency | Instructions]. This is required for backend synchronization.
4. Avoid generic advice unless clinically appropriate based on the patient's profile and symptoms.

--------------------------------------------------
CLINICAL REASONING HIERARCHY
--------------------------------------------------
1. **PROFILE AUTHORITY**: Every response must be filtered through the Patient Profile (Age, Genotype, Conditions, Allergies).
   - *Example*: If a user has the **AS Genotype** and reports fatigue in heat, you MUST explain the link between dehydration, blood viscosity, and oxygen transport.
2. **REGIONAL CONTEXT**: Use the 'Medical Database Context' (RAG) to prioritize localized diseases (Malaria, Typhoid) over general viral infections when symptoms align.
3. **DIETARY PRECISION**: For diabetic patients, analyze food impacts (e.g., Jollof Rice) based on glycemic load and fiber content as defined in your specific medical documents.

--------------------------------------------------
TECHNICAL SUBSYSTEMS
--------------------------------------------------
• **SYMPTOM TRIAGE**: Calculate internal severity (0-10) and classify urgency (EMERGENCY to LOW RISK).
• **DRUG SAFETY**: Check all suggested medications against the user's Allergies (e.g., Penicillin, Groundnut) and current Meds.
• **VISION ANALYSIS**: For food images, estimate macros; for labs, translate jargon into clinical trends.

--------------------------------------------------
EMERGENCY ESCALATION
--------------------------------------------------
If symptoms indicate a life-threatening event (chest pain, stroke signs, severe bleeding), immediately stop all general advice and instruct the user to call 112 or 767 (Lagos Emergency) and seek urgent care.

--------------------------------------------------
MISSION & LIMITATIONS
--------------------------------------------------
You are a physician-level support system, not a replacement for a doctor. Frame all advice as clinical insight or educational guidance. NEVER provide a definitive diagnosis or prescribe medication regimens.
"""

# =========================
# SMART SUMMARIZER ENDPOINT (NEW)
# =========================

@app.post("/api/generate-title")
def generate_title(request: TitleRequest):
    """Generates a short 3-word title for the sidebar."""
    if not gemini_client:
        return {"title": "New Conversation"}
    prompt = f"Summarize this medical message into exactly 3 words for a chat title: {request.message}"
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return {"title": (response.text or "").strip().replace('"', '')}
    except:
        return {"title": "New Conversation"}

# =========================
# SELF-LEARNING MEDICAL MEMORY
# =========================

def get_long_term_memory(user_id: str) -> str:
    """Fetches A.M.E.L.I.A's historical observations about the patient."""
    if not supabase or not user_id:
        return "No historical data recorded."
        
    try:
        response = supabase.table("MedicalMemory").select("observation").eq("userId", user_id).execute()
        records = response.data
        
        # Ensure records is a list before iterating
        if not records or not isinstance(records, list):
            return "No historical data recorded."
            
        # Check that 'record' is a dict to satisfy Pylance
        facts = [
            record["observation"] 
            for record in records 
            if isinstance(record, dict) and "observation" in record
        ]
        
        return "\n".join(f"- {fact}" for fact in facts)
    except Exception as e:
        print(f"Supabase fetch error: {e}")
        return "Error retrieving history."

async def extract_and_store_memory(user_id: str, user_message: str, ai_response: str):
    """Background task to extract permanent medical facts and store them."""
    if not openai_client or not user_id or not supabase: 
        return

    existing_memory = get_long_term_memory(user_id)

    extraction_prompt = f"""
    Analyze this medical exchange. Extract any NEW, permanent, or long-term medical facts about the patient.
    IGNORE temporary symptoms (e.g., a cold today) unless they indicate a chronic pattern.
    Focus on: new chronic diagnoses, regular medications, severe allergies, surgeries, or major lifestyle changes.
    
    Here is what you ALREADY know about this patient:
    {existing_memory}

    Return ONLY a valid JSON array of strings representing the facts. 
    If there is nothing new to learn, return an empty array: []

    Exchange:
    Patient: {user_message}
    A.M.E.L.I.A: {ai_response}
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a clinical data extraction bot. Output only a raw JSON array. Example: [\"Patient has a peanut allergy\", \"Patient prefers morning routines\"]"},
                {"role": "user", "content": extraction_prompt}
            ]
        )

        content = response.choices[0].message.content or "[]"
        raw_json = content.strip().strip("```json").strip("```")
        facts = json.loads(raw_json)

        for fact in facts:
            # Insert into the Prisma-generated MedicalMemory table
            supabase.table("MedicalMemory").insert({
                "id": str(uuid.uuid4()),
                "userId": user_id,
                "observation": fact
            }).execute()
            print(f"Learned new fact for {user_id}: {fact}")
            
    except Exception as e:
        print(f"Memory extraction error: {e}")

def detect_emergency(user_message):
    raise NotImplementedError

# =========================
# CHAT ENDPOINT
# =========================

@app.post("/chat")
def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    user_message = request.user_message
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id 
    profile = request.profile or {}
    image_data = request.image_data 

    # --- DYNAMIC CONTEXT & MEMORY HANDLING ---
    # Memory is now safely managed by the Next.js frontend state
    request_history = request.history or []
    
    # Fetch Long-Term Medical Memory from Supabase
    long_term_memory = get_long_term_memory(user_id) if user_id else "No long term memory available."

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
    
    --- A.M.E.L.I.A LONG-TERM MEDICAL MEMORY ---
    {long_term_memory}
    ------------------------------------
    """

    rule_emergency = detect_rule_emergency(user_message)
    urgency = triage_level(user_message)

    # 🚨 EMERGENCY HARD STOP
    if urgency == "EMERGENCY":
        return StreamingResponse(
            iter([
                "This may be a medical emergency. Please seek immediate medical attention or call emergency services immediately."
                ]),
                media_type="text/plain"
        )
    
    severity_score = calculate_symptom_score(user_message)
    context = retrieve_medical_context(user_message)
    followup_questions = generate_followup_questions(user_message)
    regional_flags = detect_regional_diseases(user_message)
    action_steps = generate_action_steps(user_message)
    med_list = profile.get("currentMeds") or []
    if isinstance(med_list, str):
         med_list = [med_list]

    drug_safety = generate_drug_safety_context(med_list, allergies)

    # Pre-process image with Gemini Vision Extractors if present
    extracted_image_context = ""
    if image_data:
        if "lab" in user_message.lower() or "test" in user_message.lower():
            extracted_image_context = extract_lab_results(image_data)
            label = "Lab Report Data"
        else:
            extracted_image_context = extract_prescription_details(image_data)
            label = "Prescription/Medication Data"

    messages: list = [ChatCompletionSystemMessageParam(role="system", content=system_prompt)]
    
    for m in request_history:
        role = m.get("role")
        if role == "user":
            messages.append(ChatCompletionUserMessageParam(role="user", content=m.get("content", "")))
        elif role == "assistant":
            messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=m.get("content", "")))

    # --- UPDATED GREETING LOGIC ---
    # is_new_session should be passed from frontend, or determined by history length
    is_new_session = request.is_new_session if request.is_new_session is not None else (len(request_history) == 0)

    if is_new_session is None:
        is_new_session = len(request_history) == 0

    greeting_instruction = f"Greet {first_name} warmly and welcome them back." if is_new_session else "DO NOT use any greetings, 'Hi', or 'Hello'. Jump straight to the medical analysis."

    followup_block = "\n".join([f"- {q}" for q in followup_questions]) if followup_questions else "None"
    action_block = "\n".join([f"- {s}" for s in action_steps]) if action_steps else "None"
    regional_block = "\n".join(regional_flags) if regional_flags else "None"

    # Build a strict hierarchy of rules
    user_prompt_text = f"""{patient_context}

    ROLE:
    You are Amelia, an AI clinical assistant trained to provide safe, structured medical guidance similar to a primary care physician.
    
    INSTRUCTION FOR THIS TURN:
    {greeting_instruction}
    ------------------------------------------
    """

    if extracted_image_context:
        user_prompt_text += f"\n--- IMAGE ANALYSIS RESULTS ({label}) ---\n{extracted_image_context}\n--------------------------------------\n"

    user_prompt_text += f"""
    --- CLINICAL REASONING SUPPORT ---

    {drug_safety}

    Possible Regional Conditions:
    {regional_block}
    
    Recommended Immediate Actions:
    {action_block}
    
    Follow-up Questions to Ask the Patient:
    {followup_block}
    
    --- RETRIEVED MEDICAL DATABASE CONTEXT ---
    {context}
    ------------------------------------------

    CRITICAL REASONING INSTRUCTIONS:
    1. **PROFILE AUTHORITY**: You MUST base your advice STRICTLY on the Patient Biometrics provided (Genotype: {genotype}, Sugar Level: {sugar_level}).
    2. **LOCALIZED TRIAGE**: If the user mentions fatigue or headache in Lagos, explain the link between AS Genotype, hydration, and heat-induced viscosity changes.
    3. **MEDICATION SYNC**: If the user confirms a dosage (e.g., 500mg Metformin), you MUST output the tag at the END of your message: [AMELIA_NEW_MED: Name | Dosage | Freq | Instructions].
    4. **SMART CONTEXT**: Use only relevant data from the 'Medical Database Context'. IGNORE irrelevant facts like 'Zenith Protocol'.
    5. **FORMATTING**: Use ONLY **Bold Text** for headers. NEVER use hashtags (#) or ****.

    Urgency level: {urgency}
    Question: {user_message}
    """

    # --- MULTIMODAL INJECTION ---
    if image_data:
        clean_img = clean_base64(image_data)
        messages.append(ChatCompletionUserMessageParam(role="user", content=[
            {"type": "text", "text": user_prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean_img}"}}
        ]))
    else:
        messages.append(ChatCompletionUserMessageParam(role="user", content=user_prompt_text))

    def generate_response():
        if not openai_client:
            yield "I'm sorry, my AI systems are currently offline."
            return

        full_response = ""
        
        try:
            # Use OpenAI for the medical route as in your previous setup
            response_stream = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True 
            )

            for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    full_response += text_chunk
                    yield text_chunk
            
            # ---> TRIGGER BACKGROUND MEMORY EXTRACTION <---
            if user_id:
                background_tasks.add_task(extract_and_store_memory, user_id, user_message, full_response)

        except Exception as e:
            print(f"Streaming Error: {e}")
            yield f"Error: {str(e)}"

    return StreamingResponse(generate_response(), media_type="text/plain")

@app.get("/")
def home():
    return {"status": "AMELIA Backend is Online", "version": "1.1.0", "environment": "local"}

# =========================
# LOCAL SERVER RUNNER
# =========================
if __name__ == "__main__":
    print("Starting A.M.E.L.I.A Local Development Server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)