import random
import time
from dotenv import load_dotenv
from typing import Optional, List, Dict
import pandas as pd
from cachetools import TTLCache
import re 
from langchain_core.documents import Document
import faiss

from pgvector.sqlalchemy import Vector




import numpy as np



#Twilio API
from twilio.rest import Client
# FastAPI & Pydantic
from fastapi import FastAPI, Response, Depends, HTTPException, Query, Path, Body, UploadFile, File
from pydantic import BaseModel
from passlib.hash import pbkdf2_sha256

# SQLAlchemy
from sqlalchemy import or_, create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean, Text, text, Float, func, ARRAY
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from sqlalchemy.dialects.postgresql import UUID, ARRAY

# Password hashing
from passlib.context import CryptContext
from werkzeug.security import generate_password_hash

# Misc
import uuid
from uuid import uuid4
from datetime import datetime, timedelta

import json
import os
import io
import math
import tempfile
import pickle
import numpy as np

# FastAPI responses & middleware
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Google APIs
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
from googleapiclient.errors import HttpError
from google.cloud import storage

# PDF & embeddings
from PyPDF2 import PdfReader
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain.vectorstores import FAISS

# Rapidfuzz for string matching
from rapidfuzz import fuzz

#global dictionary gpt maintains context in the conversation
user_contexts: dict[str, list[dict[str, str]]] = {}
MAX_INTERACTIONS = 20
interaction=0
# Global FAISS index (in-memory)
FAISS_INDEX = None
FAISS_METADATA = None
#for creating user passwords
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

 

#-------------------------------- for Twilio
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
client = Client(account_sid, auth_token)
# Temporary in-memory OTP storage
otp_store = {}
user_vectorstores_initialized = {} 
MODEL_COST = {
    "gpt-4o-mini": {"input": 0.0015, "output": 0.002},  # USD per 1k tokens
    "text-embedding-3-small": {"input": 0.0004, "output": 0},
}

# -----------------------------
# App & CORS
# -----------------------------
load_dotenv()
app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://krish-chat-bot.vercel.app",
        "https://krish-chat-bot-new.vercel.app",
        "https://chatbot.gemkidsacademy.com.au",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # or ["GET", "POST", "OPTIONS"]
    allow_headers=["*"],
)
# -----------------------------
# OpenAI Setup
# -----------------------------
openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY_S")
)




# -----------------------------
# Google Drive Setup
# -----------------------------
SERVICE_ACCOUNT_INFO = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
SCOPES = ["https://www.googleapis.com/auth/drive"]

creds = Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
drive_service = build("drive", "v3", credentials=creds)


#DEMO_FOLDER_ID = "1sWrRxOeH3MEVtc75Vk5My7MoDUk41gmf"
DEMO_FOLDER_ID = "1EweJn82tRvVD5DlHwdPKzc_uppXU5LKH"




# -----------------------------
# Google Cloud Storage Setup
# -----------------------------
# Path to your service account JSON for GCS
service_account_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if not service_account_json:
    raise ValueError("Environment variable 'GOOGLE_APPLICATION_CREDENTIALS_JSON' is missing.")

# 2️⃣ Replace literal "\n" with actual newlines for the private key
service_account_info = json.loads(service_account_json)
service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")

# 3️⃣ Initialize GCS client with credentials
gcs_client = storage.Client(
    credentials=Credentials.from_service_account_info(service_account_info),
    project=service_account_info["project_id"]
)

# 4️⃣ Access your bucket
gcs_bucket_name = "krishchatbot"
gcs_bucket = gcs_client.bucket(gcs_bucket_name)

print(f"✅ Initialized GCS client for bucket: {gcs_bucket_name}")

#---------------- database connectivity 
DATABASE_URL = os.environ.get("DATABASE_URL") or (
    f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

cached_vectorstores = TTLCache(maxsize=20, ttl=3600)
pdf_listing_done = False
all_pdfs = []

class OpenAIUsageLog(Base):
    __tablename__ = "openai_usage_log"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    model_name = Column(String)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    cost_usd = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True)
    pdf_name = Column(String, nullable=False)
    class_name = Column(String, nullable=True)
    pdf_link = Column(String, nullable=True)
    chunk_text = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_id = Column(String, nullable=False)
    embedding_vector = Column(ARRAY(Float), nullable=False)  # <-- add this
 
class GuestChatbotMessage(BaseModel):
    role: str
    content: str

class ChatRequestGuestChatbot(BaseModel):
    query: str
    context: List[GuestChatbotMessage] = []
    
class UserListItem(BaseModel):
    id: int
    name: str
    email: str
    phone_number: str
    class_name: str

    class Config:
        orm_mode = True  # allows SQLAlchemy models to be converted to Pydantic models

class KnowledgeBaseResponse(BaseModel):
    knowledge_base: str | None
    updated_at: datetime | None
        

class UserResponse(BaseModel):
    name: str
    email: str
    phone_number: str  # now required
    class_name: str    # now required
    
class UsageResponse(BaseModel):
    date: str
    amount_usd: float
    type: str

class SendOTPRequest(BaseModel):
    phone_number: str

class VerifyOTPRequest(BaseModel):
    phone_number: str
    otp: str


    
class AddUserRequest(BaseModel):
    id: int
    name: str
    email: str
    password: str
    phone_number: str
    class_name: str 

class KnowledgeBaseRequest(BaseModel):
    knowledge_base: str

class EditUserRequest(BaseModel):
    name: str
    email: str
    phone_number: str
    class_name: str
    password: Optional[str] = None  # only update if provided
    
class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class KnowledgeBaseUpdate(BaseModel):
    knowledge_base: str

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    phone_number = Column(String(20), nullable=True)  # optional
    class_name = Column(String(50), nullable=True)    # optional
    password = Column(String(255), nullable=False)    # hashed password
    created_at = Column(DateTime, default=datetime.utcnow)

    # Establish relationship with sessions
    sessions = relationship("SessionModel", back_populates="user", cascade="all, delete-orphan")


class SessionModel(Base):  # Handles authentication sessions
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_token = Column(UUID(as_uuid=True), unique=True, index=True, default=uuid.uuid4)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))  # link to User
    is_authenticated = Column(Boolean, default=False)
    public_token = Column(UUID(as_uuid=True), unique=True, index=True, default=uuid.uuid4)

    # Establish relationship back to User
    user = relationship("User", back_populates="sessions")
    
class LoginRequest(BaseModel):
    name: str
    password: str


Base.metadata.create_all(bind=engine)

ADMIN_EMAIL = "admin@example.com"
ADMIN_NAME = "Admin"
ADMIN_PASSWORD = "admin123"  # plaintext, will hash
ADMIN_PHONE = "0000000000"
ADMIN_CLASS = "Admin"

# Use a session to interact with the database
with Session(engine) as session:
    # Check if an admin user already exists
    admin_user = session.query(User).filter_by(email=ADMIN_EMAIL).first()

    if not admin_user:
        # Hash the password before storing
        hashed_password = generate_password_hash(ADMIN_PASSWORD)
        
        # Create the admin user
        new_admin = User(
            name=ADMIN_NAME,
            email=ADMIN_EMAIL,
            phone_number=ADMIN_PHONE,
            class_name=ADMIN_CLASS,
            password=hashed_password
        )
        session.add(new_admin)
        session.commit()
        print("Admin user created.")
    else:
        print("Admin user already exists.")


def get_db():
    """Yield a database session, ensuring it's closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        


# Generate OTP
def generate_otp():
    return random.randint(100000, 999999)
#log openai api usage
def log_openai_usage(
    db: Session,
    user_id: str,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float
):
    usage_entry = OpenAIUsageLog(
        user_id=user_id,
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd
    )
    db.add(usage_entry)
    db.commit()

# Send OTP SMS
def send_otp_sms(phone_number: str, otp: int):
    message = client.messages.create(
        body=f"Your OTP code is {otp}",
        from_=twilio_number,
        to=phone_number
    )
    print(f"Sent OTP {otp} to {phone_number}, SID: {message.sid}")


def calculate_openai_cost(model_name: str, prompt_tokens: int, completion_tokens: int, multiplier: float = 1.0) -> float:
    """
    Calculate approximate OpenAI API cost in USD.
    Multiplier allows for internal adjustments or markups.
    """
    model_rates = MODEL_COST.get(model_name)
    if not model_rates:
        print(f"[WARN] No cost info for model '{model_name}', defaulting to 0")
        return 0.0

    cost = (prompt_tokens / 1000) * model_rates["input"] + \
           (completion_tokens / 1000) * model_rates["output"]

    cost *= multiplier
    return round(cost, 6)
    
# -----------------------------
# API Endpoints

# -----------------------------


@app.options("/{path:path}")  # 👈 handles all OPTIONS requests
async def preflight_handler(path: str):
    """
    This ensures even if CORS middleware misses, OPTIONS is handled cleanly.
    """
    print("Received OPTIONS preflight for:", path)
    return Response(status_code=200)

@app.get("/")
async def root():
    return {"message": "Backend running with CORS enabled"}

# GET all users
@app.get("/api/users", response_model=List[UserListItem])
def get_all_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users
    
@app.get("/api/openai-usage")
def get_openai_usage(db: Session = Depends(get_db)):
    """
    Returns total OpenAI API usage per user for the last 30 days.
    Includes debug print statements for tracing.
    """
    print("\n[DEBUG] Starting /api/openai-usage endpoint")

    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    print(f"[DEBUG] Fetching usage records from {thirty_days_ago} to now")

    try:
        # Query total cost grouped by user_id for last 30 days
        usage_records = (
            db.query(
                OpenAIUsageLog.user_id,
                func.sum(OpenAIUsageLog.cost_usd).label("total_amount")
            )
            .filter(OpenAIUsageLog.timestamp >= thirty_days_ago)
            .group_by(OpenAIUsageLog.user_id)
            .all()
        )

        print(f"[DEBUG] Number of users with usage in last 30 days: {len(usage_records)}")
        
        # Convert SQLAlchemy results to JSON-friendly format
        result = []
        for record in usage_records:
            record_dict = {
                "user": record.user_id, 
                "amount_usd": float(record.total_amount or 0)
            }
            print(f"[DEBUG] User: {record.user_id}, Total Usage: ${record_dict['amount_usd']:.2f}")
            result.append(record_dict)

        print("[DEBUG] Finished preparing usage data response")
        return result  # FastAPI will automatically serialize to JSON

    except Exception as e:
        print(f"[ERROR] Failed to fetch usage data: {str(e)}")
        # Return JSON error instead of default HTML error page
        raise HTTPException(status_code=500, detail=f"Failed to fetch usage data: {str(e)}")

        

# --- Guest Chatbot endpoint ---
@app.post("/guest-chatbot")
async def guest_chatbot(request: ChatRequestGuestChatbot, db: Session = Depends(get_db)):
    print("\n==================== GUEST CHATBOT REQUEST START ====================")
    print(f"[INFO] Incoming query: {request.query}")
    print(f"[INFO] Context message count: {len(request.context)}")

    try:
        # Step 1: Load knowledge base
        print("[STEP 1] Fetching documents from knowledge_base table...")
        docs_in_db = db.execute(text("SELECT content FROM knowledge_base")).mappings().all()
        docs = [Document(page_content=row["content"]) for row in docs_in_db]

        if not docs:
            print("[WARN] Knowledge base is empty.")
            return {"snippet": "Knowledge base is empty."}

        # Step 2: Create FAISS vector store
        print("[STEP 2] Creating FAISS vector store...")
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY_S"))
        try:
            vectorstore = FAISS.from_documents(docs, embeddings)
            print("[DEBUG] Vector store created successfully.")
        except Exception as e:
            print("[ERROR] Vector store creation failed:", e)
            raise HTTPException(status_code=500, detail="Vector store creation failed")

        # Step 3: Retrieve top 3 relevant documents
        print("[STEP 3] Performing similarity search...")
        try:
            docs_and_scores = vectorstore.similarity_search_with_score(request.query, k=3)
            print(f"[DEBUG] Found {len(docs_and_scores)} relevant documents.")
        except Exception as e:
            print("[ERROR] Similarity search failed:", e)
            raise HTTPException(status_code=500, detail="Similarity search failed")

        if not docs_and_scores:
            return {"snippet": "I do not have the sufficient knowledge to answer this query."}

        combined_snippet = "\n\n".join([doc.page_content for doc, _ in docs_and_scores])

        # Step 4: Build full conversation context
        conversation_context = "\n".join(
            [f"{msg.role.capitalize()}: {msg.content}" for msg in request.context]
        )

        # Step 5: Construct prompt
        prompt = f"""
        You are an educational assistant.
        
        Answer the user's query **briefly and clearly in 2–5 sentences** using only the knowledge base below.
        If the knowledge base does not provide enough information, reply exactly:
        "I do not have the sufficient knowledge to answer this query."
        
        Knowledge Base (use only relevant content):
        {combined_snippet}
        
        Conversation so far:
        {conversation_context}
        
        Latest User Query:
        {request.query}
        """

        print("[STEP 4] Sending to OpenAI API...")
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        gpt_answer = response.choices[0].message.content.strip()

        print("[INFO] Final Answer:", gpt_answer)

        # -------------------- Log usage --------------------
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        
        # Calculate API cost
        call_cost = calculate_openai_cost("gpt-4o-mini", prompt_tokens, completion_tokens, multiplier=1.0)
        print(f"[INFO] OpenAI API cost for this call: ${call_cost}")
        
        # Save usage in DB under 'guest'
        log_openai_usage(
            db=db,
            user_id="guest",  # <-- logging as guest
            model_name="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=call_cost
        )
        print("[INFO] API usage logged in database for guest user")
        
        print("==================== GUEST CHATBOT REQUEST END ====================\n")

        return {"snippet": gpt_answer}

    except Exception as e:
        print("[FATAL ERROR]", e)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")
        


@app.post("/api/update-knowledge-base", response_model=KnowledgeBaseResponse)
def update_knowledge_base(
    data: KnowledgeBaseRequest,
    db: Session = Depends(get_db)
):
    try:
        kb = db.query(KnowledgeBase).first()
        if not kb:
            kb = KnowledgeBase(content=data.knowledge_base)
            db.add(kb)
        else:
            kb.content = data.knowledge_base
        db.commit()
        db.refresh(kb)
        return KnowledgeBaseResponse(knowledge_base=kb.content, updated_at=kb.updated_at)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update knowledge base: {e}")
    
# ----------------- GET endpoint -----------------
# GET user by ID
@app.get("/users/info/{user_id}", response_model=UserResponse)
def get_user(user_id: int = Path(..., description="ID of the user to retrieve"),
             db: Session = Depends(get_db)):
    """
    Retrieve a user's information by ID for editing (excluding password and ID).
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        name=user.name,
        email=user.email,
        phone_number=user.phone_number,
        class_name=user.class_name
    )
@app.get("/user_ids")
def get_user_ids(db: Session = Depends(get_db)):
    """
    Returns all user IDs as a list of objects:
    [{ "id": 1 }, { "id": 2 }, ...]
    """
    users = db.query(User.id).all()  # returns list of tuples like [(1,), (2,), ...]
    user_ids = [{"id": u[0]} for u in users]  # convert to list of dicts
    return user_ids

@app.delete("/delete-user/{user_id}")
def delete_user(
    user_id: int = Path(..., description="ID of the user to delete"),
    db: Session = Depends(get_db)
):
    # Fetch the user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Delete the user
    db.delete(user)
    db.commit()

    return {"message": f"User '{user.name}' deleted successfully!"}



@app.get("/api/usage", response_model=list[UsageResponse])
def get_openai_usage(days: int = 30):
    """
    Retrieve OpenAI API usage for the past `days` days using openai_client.
    Parses nested line_items to extract cost.
    """
    try:
        start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

        response = openai_client._client.get(
            f"https://api.openai.com/v1/usage?start_date={start_date}&end_date={end_date}"
        )

        data = response.json()

        if "data" not in data or not data["data"]:
            return [{"date": "N/A", "amount_usd": 0, "type": "No usage data"}]

        result = []
        for item in data["data"]:
            timestamp = item.get("aggregation_timestamp")
            date_str = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d") if timestamp else "Unknown"

            # Sum costs from line_items if available
            line_items = item.get("line_items", [])
            total_cost = sum(li.get("cost", 0) for li in line_items)

            result.append({
                "date": date_str,
                "amount_usd": round(total_cost, 4),
                "type": "usage"
            })

        return result

    except Exception as e:
        return [{"date": "N/A", "amount_usd": 0, "type": f"Error: {str(e)}"}]
        
        
@app.post("/send-otp")
def send_otp_endpoint(data: SendOTPRequest, db: Session = Depends(get_db)):
    print(f"[DEBUG] Received OTP request for phone number: {data.phone_number}")

    # Check if the phone number exists in the database
    user = db.query(User).filter(User.phone_number == data.phone_number).first()
    if not user:
        print(f"[WARNING] Phone number {data.phone_number} not found in database")
        raise HTTPException(status_code=404, detail="Phone number not found")
    
    print(f"[DEBUG] Phone number {data.phone_number} exists. User ID: {user.id}, Name: {user.name}")

    # Generate OTP
    otp = generate_otp()
    print(f"[DEBUG] Generated OTP {otp} for phone number {data.phone_number}")

    # Store OTP with expiry
    otp_store[data.phone_number] = {"otp": otp, "expiry": time.time() + 300}  # 5 min expiry
    print(f"[DEBUG] Stored OTP for {data.phone_number} with 5 min expiry")

    # Send OTP via Twilio
    try:
        print(f"[DEBUG] Attempting to send OTP to {data.phone_number} via Twilio")
        send_otp_sms(data.phone_number, otp)
        print(f"[INFO] Successfully sent OTP to {data.phone_number}")
    except Exception as e:
        print(f"[ERROR] Error sending OTP to {data.phone_number}: {e}")
        raise HTTPException(status_code=500, detail=f"Error sending SMS: {e}")

    return {"message": "OTP sent successfully"}
    
@app.post("/verify-otp")
def verify_otp(data: VerifyOTPRequest, db: Session = Depends(get_db)):
    try:
        print(f"[DEBUG] Received OTP verification request for phone number: {data.phone_number}")

        # --- Retrieve OTP record ---
        record = otp_store.get(data.phone_number)
        if not record:
            print(f"[WARNING] No OTP record found for {data.phone_number}")
            raise HTTPException(status_code=400, detail="OTP not sent")

        if time.time() > record["expiry"]:
            print(f"[WARNING] OTP for {data.phone_number} has expired")
            raise HTTPException(status_code=400, detail="OTP expired")

        if str(data.otp) != str(record["otp"]):
            print(f"[WARNING] Entered OTP ({data.otp}) does not match stored OTP ({record['otp']})")
            raise HTTPException(status_code=400, detail="Invalid OTP")

        print(f"[INFO] OTP for {data.phone_number} is valid")

        # --- Fetch user ---
        user = db.query(User).filter(User.phone_number == data.phone_number).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # --- Clear previous user state ---
        # Clear user context
        if user.name in user_contexts:
            user_contexts[user.name] = []
            print(f"[DEBUG] Cleared previous context for user {user.name}")

        # Reset per-user vectorstore flag
        user_vectorstores_initialized[user.name] = False
        print(f"[DEBUG] vectorstores_initialized for user {user.name} set to False")

        # Clear any previous OTPs for this user
        if user.phone_number in otp_store:
            del otp_store[user.phone_number]
            print(f"[DEBUG] Cleared previous OTP for {user.phone_number}")

        # Optionally: remove existing session to ensure fresh login
        existing_session = db.query(SessionModel).filter(SessionModel.user_id == user.id).first()
        if existing_session:
            db.delete(existing_session)
            db.commit()
            print(f"[DEBUG] Cleared existing session for user {user.id}")

        # --- Clear OTP after successful verification ---
        otp_store.pop(data.phone_number, None)
        print(f"[DEBUG] Cleared OTP for {data.phone_number} after successful verification")

        # --- Prepare response ---
        user_info = {
            "id": user.id,
            "name": user.name,
            "phone_number": user.phone_number,
            "class_name": user.class_name,
        }

        print(f"[INFO] OTP verification complete for user {user.name}")
        return {"message": "OTP verified successfully", "user": user_info}

    except HTTPException as e:
        # Re-raise HTTPExceptions so FastAPI handles them normally
        raise e

    except Exception as e:
        # Catch-all for unexpected errors
        print(f"[ERROR] Unexpected exception during OTP verification for {data.phone_number}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during OTP verification")


@app.post("/login")
async def login(
    login_request: LoginRequest,
    response: Response,
    db: Session = Depends(get_db)
):
    print("\n==================== LOGIN ATTEMPT START ====================")
    global user_contexts, user_vectorstores_initialized 

    # Step 1: Log the incoming data
    try:
        print("DEBUG: Incoming request:", login_request.dict())
    except Exception as e:
        print("ERROR: Could not print login_request:", e)

    name = login_request.name
    password = login_request.password

    if not name or not password:
        print("ERROR: Missing name or password.")
        raise HTTPException(status_code=400, detail="Name and password are required")

    # Step 2: Fetch user from DB
    print("\n--- Step 2: Fetching user from DB ---")
    try:
        user = db.query(User).filter(User.name == name).first()
        if user:
            print(f"DEBUG: Found user -> ID={user.id}, Name={user.name}, Class={user.class_name}")
        else:
            print("DEBUG: No user found with that name.")
    except Exception as e:
        print("ERROR while querying user:", e)
        raise HTTPException(status_code=500, detail="Database query failed")

    if not user:
        print("==================== LOGIN ATTEMPT END (invalid name) ====================\n")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Step 3: Verify password
    print("\n--- Step 3: Verifying password ---")
    try:
        from werkzeug.security import check_password_hash
    
        print("DEBUG: Stored password in DB:", user.password)
        print("DEBUG: Password provided by user:", password)
        
        password_verified = user.password == password  # plain comparison
        print("DEBUG: Password verification result:", password_verified)
    except Exception as e:
        print("ERROR during password verification:", e)
        raise HTTPException(status_code=500, detail="Password verification failed")
    
    if not password_verified:
        print("==================== LOGIN ATTEMPT END (invalid password) ====================\n")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    print("DEBUG: Password verified successfully.")
    # Step 4: Check for existing session
    print("\n--- Step 4: Managing session ---")
    try:
        existing_session = db.query(SessionModel).filter(SessionModel.user_id == user.id).first()
        if existing_session:
            print(f"DEBUG: Existing session found for user {user.id}")
            session_token = existing_session.session_token
            public_token = existing_session.public_token
            if user.name in user_contexts:
                user_contexts[user.name] = []
                print(f"DEBUG: Cleared previous context for user {user.name}")
            # Clear previous OTP for this user
            if user.phone_number in otp_store:
                del otp_store[user.phone_number]
                print(f"DEBUG: Cleared previous OTP for {user.phone_number}")

        else:
            print("DEBUG: No session found, creating a new one...")
            session_token = str(uuid4())
            public_token = str(uuid4())
            new_session = SessionModel(
                session_token=session_token,
                public_token=public_token,
                user_id=user.id
            )
            db.add(new_session)
            db.commit()
            print("DEBUG: New session created successfully.")

            
    except Exception as e:
        print("ERROR during session handling:", e)
        raise HTTPException(status_code=500, detail="Session handling failed")

    # --- Reset in-memory context and vectorstore flags ---
    
    
    # Clear old context memory
    user_contexts[user.name] = []
    print(f"[DEBUG] Cleared in-memory chat context for {user.name}")
    
    # Reset vectorstore initialization
    user_vectorstores_initialized[user.id] = False
    print(f"[DEBUG] Reset vectorstore initialization flag for user {user.id}")
    
    # (Optional) If you have other memory states like conversation_gists:
    if "conversation_gists" in globals():
        conversation_gists[user.id] = ""
        print(f"[DEBUG] Cleared context gist for user {user.id}")


    # Step 5: Set session cookie
    print("\n--- Step 5: Setting cookie ---")
    try:
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            secure=False,  # set to True if using HTTPS
            samesite="Lax",
            max_age=3600
        )
        print(f"DEBUG: Cookie set with session_token={session_token}")
    except Exception as e:
        print("ERROR while setting cookie:", e)

    # Step 6: Prepare response
    print("\n--- Step 6: Preparing response ---")
    response_content = {
        "message": "Login successful",
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "class_name": user.class_name,       # added class_name
        "session_token": str(session_token),  # convert to string
        "public_token": str(public_token)     # convert to string
    }

    print("DEBUG: Response content:", response_content)
    print("==================== LOGIN ATTEMPT SUCCESS ====================\n")

    return JSONResponse(content=response_content, status_code=200)

@app.get("/get_next_user_id")
def get_next_user_id(db: Session = Depends(get_db)):
    last_user = db.query(User).order_by(User.id.desc()).first()
    next_id = (last_user.id + 1) if last_user else 1
    return next_id



def give_drive_access(file_id: str, emails: str, role: str = "reader"):
    """
    Grants access to a Google Drive file/folder for a list of emails.
    
    :param file_id: ID of the Drive file/folder
    :param emails: Comma-separated string of emails
    :param role: "reader" or "writer"
    """
    # Split and clean the email list
    email_list = [email.strip() for email in emails.split(",") if email.strip()]
    
    for email in email_list:
        try:
            drive_service.permissions().create(
                fileId=file_id,
                body={
                    "type": "user",
                    "role": role,
                    "emailAddress": email
                },
                fields="id",
                sendNotificationEmail=True  # optional: sends email notification to user
            ).execute()
            print(f"Access granted to {email}")
        except HttpError as error:
            print(f"Failed to give access to {email}: {error}")

@app.post("/add_user")
def add_user(user_request: AddUserRequest, db: Session = Depends(get_db)):
    # Check if email already exists
    existing_user = db.query(User).filter(User.email == user_request.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash the password using Werkzeug (compatible with check_password_hash later)
    hashed_password = generate_password_hash(user_request.password)

    # Create new user instance
    new_user = User(
        
        name=user_request.name,
        email=user_request.email,
        phone_number=user_request.phone_number,
        class_name=user_request.class_name,
        password=hashed_password,
        created_at=datetime.utcnow()
    )

    # Save to DB
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # ---------- Grant Google Drive Access ----------
    
    give_drive_access(DEMO_FOLDER_ID, user_request.email, role="reader")

    return {"message": f"User '{new_user.name}' added successfully and Drive access granted!"}

@app.put("/edit-user/{user_id}")
def edit_user(
    user_id: int = Path(..., description="ID of the user to update"),
    user_request: EditUserRequest = Body(...),
    db: Session = Depends(get_db)
):
    # Fetch the user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if email is being updated and is unique
    if user.email != user_request.email:
        existing_user = db.query(User).filter(User.email == user_request.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

    # Update fields
    user.name = user_request.name
    user.email = user_request.email
    user.phone_number = user_request.phone_number
    user.class_name = user_request.class_name

    # Only update password if provided
    if user_request.password:
        user.password = generate_password_hash(user_request.password)

    user.updated_at = datetime.utcnow()  # optional: track updates

    db.commit()
    db.refresh(user)

    return {"message": f"User '{user.name}' updated successfully!"}

        

#----------------------functions

def upload_to_gcs(file_bytes, blob_name):
    blob = gcs_bucket.blob(blob_name)
    blob.upload_from_string(file_bytes)
    print(f"Uploaded {blob_name} to bucket {gcs_bucket.name}")

def download_from_gcs(blob_name):
    blob = gcs_bucket.blob(blob_name)
    return blob.download_as_bytes()

# -----------------------------
# PDF Utilities
# -----------------------------
def download_pdf(file_id):
    fh = io.BytesIO()
    request = drive_service.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read()

def list_pdfs(folder_id, path=""):
    """
    Recursively list PDFs from Google Drive and track folder path.
    Returns a list of dicts with 'id', 'name', 'webViewLink', and 'path'.
    Includes detailed debug print statements to find root cause of empty results.
    """
    results = []
    page_token = None

    print(f"DEBUG: Starting to list files in folder_id='{folder_id}' with path='{path}'")

    while True:
        try:
            response = drive_service.files().list( 
            q=f"'{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType, webViewLink, owners(emailAddress))',
            pageToken=page_token,
            includeItemsFromAllDrives=True,  # required for shared folders
            supportsAllDrives=True,          # required for shared folders
            corpora='user'                   # crucial for shared folders visibility
        ).execute()
        
            print("\n=== DEBUG INFO ===")
            print(f"Folder ID being listed: {folder_id}")
            print(f"Number of files returned: {len(response.get('files', []))}")
            for f in response.get('files', []):
                print(f" - {f['name']} ({f['id']}) | Owner: {f.get('owners', [{}])[0].get('emailAddress')}")
            print("===================")
        except Exception as e:
            print(f"[ERROR] Failed to list files in folder_id='{folder_id}': {e}")
            return results

        files = response.get('files', [])
        print(f"DEBUG: Retrieved {len(files)} files from folder_id='{folder_id}'")

        if len(files) == 0:
            print(f"[WARNING] No files found in folder_id='{folder_id}'. Possible causes:")
            print("  - Folder ID might be wrong")
            print("  - Service account / credentials lack permission")
            print("  - Folder is empty")
            print("  - Files are not accessible or trashed")

        for file in files:
            file_id = file.get('id', '<missing_id>')
            file_name = file.get('name', '<missing_name>')
            mime_type = file.get('mimeType', '<missing_mime>')
            web_view_link = file.get('webViewLink', '')

            print(f"DEBUG: Inspecting file -> id: {file_id}, name: '{file_name}', mimeType: '{mime_type}'")

            if mime_type == 'application/pdf':
                pdf_path = f"{path}/{file_name}".lstrip("/")
                results.append({
                    "id": file_id,
                    "name": file_name,
                    "webViewLink": web_view_link,
                    "path": pdf_path
                })
                print(f"[FOUND PDF] id: {file_id}, name: '{file_name}', path: '{pdf_path}'")

            elif mime_type == 'application/vnd.google-apps.folder':
                folder_path = f"{path}/{file_name}".lstrip("/")
                print(f"[FOUND FOLDER] id: {file_id}, name: '{file_name}', path: '{folder_path}'")
                nested_results = list_pdfs(file_id, folder_path)
                results.extend(nested_results)
                print(f"[COMPLETED FOLDER] '{folder_path}', found {len(nested_results)} PDFs inside")

            else:
                print(f"[SKIPPED] id: {file_id}, name: '{file_name}', mimeType: '{mime_type}'")

        page_token = response.get('nextPageToken', None)
        if page_token:
            print(f"DEBUG: Next page token detected, continuing listing for folder_id='{folder_id}'")
        else:
            print(f"DEBUG: No more pages in folder_id='{folder_id}', finishing listing")

        if not page_token:
            break

    print(f"DEBUG: Finished listing folder_id='{folder_id}', total PDFs found so far: {len(results)}")
    return results




    
def summarize_with_openai(snippet: str, query: str) -> str:
    prompt = f"""
    You are an educational assistant.

    Based on the following PDF content:
    {snippet}

    Answer concisely what this PDF is about in the context of the question: '{query}'.
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# -----------------------------
# Vector Store Utilities
# -----------------------------
MAX_CHARS_PER_SNIPPET = 4000
THRESHOLD = 75

def vectorstore_exists_in_drive(pdf_name, folder_id):
    vs_name = f"{pdf_name}.pkl"
    response = drive_service.files().list(
        q=f"'{folder_id}' in parents and name='{vs_name}' and trashed=false",
        fields="files(id, name)"
    ).execute()
    return bool(response.get("files", []))

def upload_vectorstore_to_drive(vectorstore_bytes, vs_name, folder_id):
    file_metadata = {"name": vs_name, "parents": [folder_id]}
    from googleapiclient.http import MediaIoBaseUpload
    media = MediaIoBaseUpload(io.BytesIO(vectorstore_bytes), mimetype="application/octet-stream")
    drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()




# Keep track of PDFs processed in this run
processed_pdfs = set()

def create_vectorstore_for_pdf(pdf_file):
    """
    Process a single PDF from Google Drive, create embeddings using OpenAI's
    'text-embedding-3-large', save a FAISS vector store in GCS, and prepare metadata
    for DB upload with numeric embeddings.

    Key improvements:
    1. Metadata always includes pdf_name, page_number, chunk_index, pdf_link, and chunk_text.
    2. Embeddings are numeric arrays compatible with DB Vector column.
    3. Empty pages/chunks are skipped with debug logs.
    4. Optional L2 normalization is consistent.
    """

    pdf_name = pdf_file.get("name", "Unknown")
    pdf_id = pdf_file.get("id")
    pdf_path = pdf_file.get("path", pdf_name)
    pdf_link = pdf_file.get("webViewLink", "")

    print(f"[DEBUG] Starting vector store creation for PDF: {pdf_name}")

    # Skip PDFs with no Drive ID
    if not pdf_id:
        print(f"[DEBUG] PDF {pdf_name} has no Drive ID. Skipping.")
        return

    # Skip PDFs already in GCS
    if gcs_bucket.blob(pdf_path).exists():
        print(f"[DEBUG] PDF {pdf_name} already exists in GCS. Skipping upload.")
        return

    # Step 1: Download PDF
    try:
        file_bytes = download_pdf(pdf_id)
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        print(f"[ERROR] Failed to download or read PDF {pdf_name}: {e}")
        return

    # Step 2: Split PDF into text chunks
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text()
            if not page_text or not page_text.strip():
                continue
            cleaned_text = " ".join(line.strip() for line in page_text.splitlines() if line.strip())

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=250,
                separators=["\n\n", "\n", " "]
            )
            page_chunks = text_splitter.create_documents([cleaned_text])
            # Determine class_name robustly from pdf_path
            path_parts = pdf_path.split("/")
            class_name = next(
                (part for part in path_parts if part.lower().startswith("year") or "kindergarten" in part.lower()),
                "Unknown"
            )


            for chunk_idx, chunk in enumerate(page_chunks, start=1):
                # Ensure metadata is complete
                chunk.metadata.update({
                    "pdf_name": pdf_name,
                    "class_name": class_name,  # ✅ added
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                    "pdf_link": pdf_link,
                    "chunk_text": chunk.page_content
                })
                chunks.append(chunk)
        except Exception as e:
            print(f"[ERROR] Failed to process page {page_num} of PDF {pdf_name}: {e}")

    if not chunks:
        print(f"[DEBUG] No text chunks found in PDF {pdf_name}. Skipping vector store creation.")
        return

    # Step 3: Create embeddings
    try:
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.environ.get("OPENAI_API_KEY_S")
        )
        vs = FAISS.from_documents(chunks, embeddings_model)

        if hasattr(vs.index, "normalize_L2"):
            vs.index.normalize_L2()  # optional: consistent similarity
        print(f"[DEBUG] Vector store created for PDF: {pdf_name}")
    except Exception as e:
        print(f"[ERROR] Failed to create embeddings/vector store for PDF {pdf_name}: {e}")
        return

    # Step 4: Save vector store to GCS
    try:
        pdf_base_name = pdf_name.rsplit('.', 1)[0]
        gcs_prefix_vs = f"{os.path.dirname(pdf_path)}/vectorstore_{pdf_base_name}/"

        with tempfile.TemporaryDirectory() as tmp_dir:
            vs.save_local(tmp_dir)

            for root, dirs, files in os.walk(tmp_dir):
                for filename in files:
                    path = os.path.join(root, filename)
                    relative_path = os.path.relpath(path, tmp_dir)
                    blob_name = f"{gcs_prefix_vs}{relative_path.replace(os.sep, '/')}"
                    upload_to_gcs(open(path, "rb").read(), blob_name)
                    print(f"[DEBUG] Uploaded vector store file to GCS: {blob_name}")
    except Exception as e:
        print(f"[ERROR] Failed to upload vector store for PDF {pdf_name} to GCS: {e}")
        return

    # Step 5: Upload original PDF
    try:
        upload_to_gcs(file_bytes, pdf_path)
        print(f"[INFO] Uploaded PDF and vector store for {pdf_name} to GCS.")
    except Exception as e:
        print(f"[ERROR] Failed to upload PDF {pdf_name} to GCS: {e}")






processed_pdfs = set()  # Keep track of PDFs processed in this session

def ensure_vectorstores_for_all_pdfs(pdf_files):
    """
    Ensures vector stores are created for all PDFs in the provided list.
    Uses GCS existence checks and session memory to avoid duplicate work.
    """
    for pdf in pdf_files:
        pdf_id = pdf.get("id")
        pdf_name = pdf.get("name", "Unknown")
        pdf_path = pdf.get("path")  # Full folder path from Drive

        if not pdf_id:
            print(f"[DEBUG] PDF {pdf_name} has no Drive ID. Skipping.")
            continue

        # Skip if already processed in this session
        if pdf_id in processed_pdfs:
            print(f"[DEBUG] PDF {pdf_name} already processed in this session. Skipping.")
            continue

        # Skip if PDF already exists in GCS
        if gcs_bucket.blob(pdf_path).exists():
            print(f"[DEBUG] PDF {pdf_name} already exists in GCS. Skipping.")
            processed_pdfs.add(pdf_id)
            continue

        # Print before creating vector store
        print(f"[DEBUG] Creating vector store for PDF: {pdf_name}, Path: {pdf_path}")

        # Create vector store for this PDF
        create_vectorstore_for_pdf(pdf)

        # Print after successful creation
        print(f"[DEBUG] Vector store created for PDF: {pdf_name}")

        # Mark as processed in memory
        processed_pdfs.add(pdf_id) 
        
def load_vectorstore_from_gcs(gcs_prefix: str, embeddings: OpenAIEmbeddings) -> FAISS:
    """
    Downloads the FAISS vector store files from a GCS prefix and loads the vector store.

    Args:
        gcs_prefix (str): The folder/prefix in GCS where the vector store files are located.
        embeddings (OpenAIEmbeddings): The embeddings object to use for loading the vector store.

    Returns:
        FAISS: The loaded FAISS vector store instance.
    """
    print(f"\n[DEBUG][GCS-LOAD] ======== Starting load_vectorstore_from_gcs ========")
    print(f"[DEBUG][GCS-LOAD] Prefix: {gcs_prefix}")
    print(f"[DEBUG][GCS-LOAD] Embeddings type: {type(embeddings)}")
    print(f"[DEBUG][GCS-LOAD] Embeddings model: {getattr(embeddings, 'model', 'Unknown')}")

    try:
        # Ensure GCS client and bucket are available
        if "gcs_client" not in globals() or "gcs_bucket_name" not in globals():
            raise RuntimeError("GCS client or bucket name not initialized in globals()")

        print(f"[DEBUG][GCS-LOAD] Using bucket: {gcs_bucket_name}")

        # List all blobs under the given prefix
        blobs = list(gcs_client.list_blobs(gcs_bucket_name, prefix=gcs_prefix))
        print(f"[DEBUG][GCS-LOAD] Found {len(blobs)} blobs for prefix '{gcs_prefix}'")

        if not blobs:
            raise ValueError(f"No vector store files found in GCS for prefix '{gcs_prefix}'.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f"[DEBUG][GCS-LOAD] Created temporary directory: {tmp_dir}")

            # Download each blob
            for blob in blobs:
                file_path = os.path.join(tmp_dir, os.path.basename(blob.name))
                print(f"[DEBUG][GCS-LOAD] Downloading: {blob.name} → {file_path}")
                blob.download_to_filename(file_path)

            # Verify downloaded files
            downloaded_files = os.listdir(tmp_dir)
            print(f"[DEBUG][GCS-LOAD] Downloaded files: {downloaded_files}")

            # Load the FAISS vectorstore
            print(f"[DEBUG][GCS-LOAD] Loading FAISS index from temp dir...")
            vs = FAISS.load_local(
                tmp_dir,
                embeddings,
                allow_dangerous_deserialization=True  # safe for trusted environment
            )

            print(f"[DEBUG][GCS-LOAD] Successfully loaded FAISS store.")
            print(f"[DEBUG][GCS-LOAD] ======== Finished load_vectorstore_from_gcs ========\n")

            return vs

    except Exception as e:
        import traceback
        print(f"[ERROR][GCS-LOAD] Exception in load_vectorstore_from_gcs(): {type(e).__name__} - {e}")
        print("[ERROR][GCS-LOAD] Full traceback:")
        traceback.print_exc()
        raise  # Re-raise to surface the real cause
    
SIMILARITY_THRESHOLD = 0.40  # cosine similarity threshold (adjust as needed)
TOP_K = 5  # max chunks per PDF
REWRITER_MODEL = "gpt-4o-mini"
ANSWER_MODEL = "gpt-4o-mini"
  # user_id -> list of messages
GIST_MAX_LENGTH = 1000

def get_context_gist(user_id, max_tokens=1000):
    """
    Returns previous conversation including PDF metadata for GPT context
    """
    context_entries = user_contexts.get(user_id, [])
    gist = []
    for entry in context_entries:
        gist.append(f"{entry['role'].capitalize()}: {entry['content']}")
    return "\n".join(gist)
    
#classify query type
def classify_query_type(
    query: str, 
    context_gist: str, 
    user_id: str, 
    pdf_list: list,  # new: list of available PDFs for the user
    db: Session  # remove Depends
) -> str:
    """
    Classify if a query is 'context_only', 'pdf_only', or 'mixed'.
    Now takes PDF list into account.
    Logs OpenAI usage against the user_id if db session is provided.
    """
    REWRITER_MODEL = "gpt-4o-mini"

    # Prepare a short summary of PDFs
    pdf_summary = "\n".join([f"- {pdf['name']}" for pdf in pdf_list[:50]])  # limit to 50 PDFs

    prompt = f"""
You are a classifier assistant. 
Determine whether the following user query requires:
1. Only previous conversation context to answer (context_only)
2. Only new information from PDF resources (pdf_only)
3. Both previous context and PDFs (mixed)

Previous Context (gist):
{context_gist}

Available PDFs (titles / topics):
{pdf_summary}

User Query:
{query}

Instructions:
- If the answer could exist in any of the PDFs above, classify as 'pdf_only' or 'mixed' appropriately.
- If it can only be answered from previous conversation context, classify as 'context_only'.
- Respond with exactly one word: context_only, pdf_only, or mixed.
"""

    print(f"[STEP] Sending classification request to OpenAI for user_id={user_id}")
    response = openai_client.chat.completions.create(
        model=REWRITER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    gpt_classification = response.choices[0].message.content.strip()
    print(f"[INFO] Classification result for user_id={user_id}: {gpt_classification}")

    # -------------------- Log usage --------------------
    if db:
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        call_cost = calculate_openai_cost(REWRITER_MODEL, prompt_tokens, completion_tokens, multiplier=1.0)
        print(f"[INFO] OpenAI API cost for classification call: ${call_cost}")

        log_openai_usage(
            db=db,
            user_id=user_id,
            model_name=REWRITER_MODEL,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=call_cost
        )
        print(f"[INFO] API usage logged for user_id={user_id}")

    # -------------------- Safety fallback --------------------
    # If user requested a class_name / PDFs explicitly, ensure at least 'pdf_only'
    if gpt_classification == "context_only" and pdf_list:
        gpt_classification = "pdf_only"
        print(f"[INFO] Fallback applied: query contains PDFs, forcing 'pdf_only'")

    return gpt_classification


def append_to_user_context(user_id, role, content, pdf_meta=None):
    """
    pdf_meta: list of strings like '[PDF used: filename (Page X)]'
    """
    if user_id not in user_contexts:
        user_contexts[user_id] = []

    entry = {"role": role, "content": content}
    if pdf_meta:
        entry["content"] += "\n" + "\n".join(pdf_meta)
    user_contexts[user_id].append(entry)


def gcs_vectorstore_exists(prefix: str) -> bool:
    """
    Check if a vectorstore folder exists in GCS for the given prefix.
    Returns True if any files exist under that prefix.
    """
    blobs = list(gcs_bucket.list_blobs(prefix=prefix))
    print(f"[DEBUG] Checking GCS vectorstore existence for prefix: '{prefix}', found {len(blobs)} files")
    return len(blobs) > 0

def get_vectorstore_from_cache(gcs_prefix: str, embeddings):
    global cached_vectorstores

    if gcs_prefix not in cached_vectorstores:
        print(f"[CACHE MISS] Loading vectorstore from GCS: {gcs_prefix}")
        vectorstore = load_vectorstore_from_gcs(gcs_prefix, embeddings)
        cached_vectorstores[gcs_prefix] = vectorstore
    else:
        print(f"[CACHE HIT] Using cached vectorstore: {gcs_prefix}")
        vectorstore = cached_vectorstores[gcs_prefix]

    return vectorstore  


def is_pdf_request(
    query: str, 
    user_id: str, 
    db: Session
) -> bool:
    """
    Determine whether the user wants PDF links.
    Returns True if the model answers 'YES' or if query matches a term/week pattern.
    Logs API usage if db session is provided.
    """
    query_lower = query.lower()

    # -------------------- Pattern-based check --------------------
    term_week_pattern = r"term\s*\d+\s*week\s*\d+"
    if re.search(term_week_pattern, query_lower):
        return True

    # -------------------- OpenAI classifier --------------------
    prompt = (
        "You are a strict intent classifier.\n\n"
        "Determine whether the following user query is asking to fetch PDF files or PDF links.\n"
        "Respond only with YES or NO (without quotes).\n\n"
        f"Query: {query}"
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        # -------------------- Log usage --------------------
        if db is not None:
            try:
                usage = response.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens

                call_cost = calculate_openai_cost(
                    "gpt-4o-mini", prompt_tokens, completion_tokens, multiplier=1.0
                )

                log_openai_usage(
                    db=db,
                    user_id=user_id,
                    model_name="gpt-4o-mini",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=call_cost
                )
                print(f"[INFO] OpenAI API usage logged for user {user_id}, cost: ${call_cost}")
            except Exception as e:
                print(f"[WARN] Failed to log API usage for is_pdf_request: {e}")

        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("YES")

    except Exception as e:
        print(f"[ERROR] is_pdf_request failed: {e}")
        # fallback: treat as not a PDF request
        return False




def generate_drive_pdf_url(file_id: str) -> str:
        return f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"

def is_educational_query_openai(query: str, user_id: str, db: Session) -> bool:
    """
    Returns True if OpenAI classifies the query as educational, False otherwise.
    Logs OpenAI API usage to the database.
    """
    # -------------------- Quick check for 'year' or 'term' --------------------
    if any(word in query.lower() for word in ["year", "term"]):
        return True
    # -------------------- Prepare prompt --------------------
    prompt = (
        "You are a helpful assistant that classifies queries as educational or not.\n\n"
        "Determine if the following query is educational, i.e., related to school subjects, "
        "lessons, exercises, assignments, quizzes, or academic content. Respond ONLY with Yes or No.\n\n"
        f"Query: \"{query}\""
    )

    # -------------------- Call OpenAI --------------------
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content.strip().lower()
    is_educational = answer.startswith("yes")

    # -------------------- Log usage --------------------
    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)

        # Calculate API cost
        call_cost = calculate_openai_cost(
            model_name="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            multiplier=1.0
        )
        print(f"[INFO] OpenAI API cost for this call: ${call_cost}")

        # Save usage in DB for the user
        log_openai_usage(
            db=db,
            user_id=user_id,
            model_name="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=call_cost
        )
        print(f"[INFO] API usage logged in database for user_id={user_id}")

    return is_educational


embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY_S")
)

def search_top_k(query_text: str, top_k: int = 5, class_filter: list = None):
    """
    Perform fast similarity search in-memory using FAISS.

    Args:
        query_text (str): User query.
        top_k (int): Number of results to return.
        class_filter (list): Optional list of class_names to filter results.

    Returns:
        List of metadata dicts with pdf_name, class_name, chunk_index, similarity score.
    """
    # 1. Embed the query
    query_vector = embedding_model.embed_query(query_text)
    query_vector = np.array([query_vector], dtype='float32')
    faiss.normalize_L2(query_vector)

    # 2. Search FAISS
    distances, indices = index.search(query_vector, top_k * 3)  # search more if you plan to filter by class

    # 3. Map to metadata and optionally filter by class
    results = []
    for i, score in zip(indices[0], distances[0]):
        meta = metadata[i]
        if class_filter:
            if not any(cf.lower().strip() in meta['class_name'].lower() for cf in class_filter):
                continue
        results.append({**meta, "score": float(score)})
        if len(results) >= top_k:
            break

    return results
    

@app.get("/search")
async def search_pdfs(
    query: str,
    reasoning: str,
    user_id: str,
    class_name: str = None,
    db: Session = Depends(get_db)
):
    print("\n==================== SEARCH REQUEST START ====================")
    print(f"[INFO] user_id: {user_id}, query: {query}, reasoning: {reasoning}, class_name: {class_name}")
    global FAISS_INDEX, FAISS_METADATA

    results = []
    top_chunks = []
    answer_text = ""  # initialize here to append PDF metadata later
    global user_contexts, user_vectorstores_initialized, pdf_listing_done, all_pdfs

    # ------------------ Step 0: Interaction limit ------------------
    user_contexts.setdefault(user_id, [])
    interaction_count = len(user_contexts[user_id]) // 2

    if interaction_count >= MAX_INTERACTIONS:
        return JSONResponse([{
            "name": "**Academy Answer**",
            "snippet": f"You have reached the maximum of {MAX_INTERACTIONS} interactions. Please contact support for further queries.",
            "links": []
        }])

    # ------------------ Step 0b: Check educational query ------------------
    if not is_educational_query_openai(query, user_id=user_id, db=db):
        return JSONResponse([{
            "name": "**Academy Answer**",
            "snippet": "Your query does not seem to be educational or relevant.",
            "links": []
        }])

    # ------------------ Step 1: Prepare PDF list ------------------
    if not pdf_listing_done:
        print("[INFO] Fetching PDF list from Google Drive for the first time...")
        all_pdfs = list_pdfs(DEMO_FOLDER_ID)
        pdf_listing_done = True

    # Normalize class names
    

    def normalize_path(path: str) -> str:
        path = path.lower().strip()
        path = path.replace("  ", " ")
        path = path.replace(".pdf.pdf", ".pdf")
        path = path.replace("- ", "-")
        return path
     
    class_names_list = [cn.strip().lower() for cn in class_name.split(",")] if class_name else []

    pdf_files = []
    for pdf in all_pdfs:
        pdf_path_norm = normalize_path(pdf.get("path", ""))
    
        # If no class filter → accept ALL PDFs
        if not class_names_list:
            pdf_files.append(pdf)
            continue
    
        # Flexible class matching using substring
        if any(cls in pdf_path_norm for cls in class_names_list):
            pdf_files.append(pdf)


    context_gist = get_context_gist(user_id)
    is_first_query = len(user_contexts[user_id]) == 0

    # ------------------ Step 2: Classify query ------------------
    query_type = classify_query_type(query, context_gist, user_id, pdf_files, db=db)
    if is_first_query:
        query_type = "pdf_only"

    use_context_only = query_type == "context_only"

    # ------------------ Step 3: Handle PDF link requests ------------------
    pdf_urls_to_send = []
    if pdf_files and is_pdf_request(query, user_id=user_id, db=db):
        query_lower = query.lower()
        year_match = re.search(r"year\s*(\d+)", query_lower)
        query_year = year_match.group(1) if year_match else None
        term_match = re.search(r"term\s*(\d+)", query_lower)
        query_term = term_match.group(1) if term_match else None
        week_match = re.search(r"week\s*(\d+)", query_lower)
        query_week = week_match.group(1) if week_match else None

        filtered_pdfs = []
        for pdf in pdf_files:
            path_lower = pdf.get("path", "").lower()
            name_lower = pdf["name"].lower()
            if query_year and not (
                f"year {query_year}" in path_lower or
                f"year_{query_year}" in path_lower or
                f"y{query_year}" in name_lower
            ):
                continue
            term_matches = re.findall(r"(?:term|t)\s*[_-]*\s*(\d+)", name_lower)
            week_matches = re.findall(r"(?:week|w)\s*[_-]*\s*(\d+)", name_lower)
            if (query_term is None or query_term in term_matches) and (query_week is None or query_week in week_matches):
                filtered_pdfs.append(pdf)

        if filtered_pdfs:
            pdf_urls_to_send = [generate_drive_pdf_url(pdf["id"]) for pdf in filtered_pdfs]
            answer_text = f"Here are the PDFs you requested:\n" + "\n".join(pdf_urls_to_send)
        else:
            answer_text = "No PDFs found."

        source_name = "Academy Answer"
        results.append({
            "name": f"**{source_name}**",
            "snippet": answer_text,
            "links": pdf_urls_to_send
        })
        append_to_user_context(user_id, "user", query)
        append_to_user_context(user_id, "assistant", answer_text)
        return JSONResponse(results)

    # ------------------ Step 4: Retrieve top PDF chunks ------------------
    if pdf_files and not use_context_only:
        class_list = [cn.strip() for cn in class_name.split(",")] if class_name else []
        if FAISS_INDEX is not None and FAISS_METADATA is not None:
            # Step 1a: Convert query to embedding
            query_embedding = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=os.environ.get("OPENAI_API_KEY_S")
            ).embed_query(query)
        
            query_vector = np.array([query_embedding], dtype='float32')
            faiss.normalize_L2(query_vector)
        
            # Step 1b: Search FAISS index
            k = TOP_K
            D, I = FAISS_INDEX.search(query_vector, k)  # distances, indices
        
            top_chunks = []
            for idx, score in zip(I[0], D[0]):
                if idx >= len(FAISS_METADATA):
                    continue
                chunk_meta = FAISS_METADATA[idx].copy()
                chunk_meta['score'] = float(score)
                top_chunks.append(chunk_meta)
        
            # Optional: Filter by class_name if provided
            if class_list:
                top_chunks = [c for c in top_chunks if c.get("class_name") in class_list]
        
            # Sort by score descending
            top_chunks = sorted(top_chunks, key=lambda x: x['score'], reverse=True)[:TOP_K]
        else:
            top_chunks = []  # fallback if FAISS not initialized

     
        if not top_chunks:
            use_context_only = True

        

        # Build context string
        context_texts_str = "\n".join(
            f"PDF: {c.get('pdf_name','N/A')} (Page {c.get('page_number','N/A')})\n{c.get('chunk_text','')}"
            for c in top_chunks
        )

    # ------------------ Step 5: Prepare GPT prompt ------------------
    reasoning_instruction = {
        "simple": "Use plain, beginner-friendly language. Keep sentences short and avoid jargon.",
        "medium": "Give a balanced explanation — clear, moderately detailed, and easy to follow.",
        "advanced": "Provide a detailed, analytical, and example-rich explanation that shows expert understanding."
    }.get(reasoning, "Use plain, beginner-friendly language.")

    if use_context_only or not top_chunks:
        gpt_prompt = f"""
You are an assistant. Follow the instructions below carefully.

Style: {reasoning_instruction}

Use only the previous conversation context to answer:
{context_gist}

Question:
{query}

Guidelines:
- Do not use any external knowledge beyond the conversation.
- If the user asks for a resource mentioned but not provided, say you do not have access.
- Prepend "[GPT answer]" if relying on your own understanding.
"""
    else:
        gpt_prompt = f"""
You are an assistant. Follow the instructions below carefully.

Style: {reasoning_instruction}

Previous conversation context:
{context_gist}

PDF Chunks:
{context_texts_str}

Question:
{query}

Guidelines:
1. Use PDF chunks if relevant.
2. Do not invent facts.
3. Prepend "[PDF-based answer]" if using PDFs, else "[GPT answer]".
"""

    # ------------------ Step 6: Call GPT ------------------
    answer_response = openai_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": gpt_prompt}],
        temperature=0.2
    )
    answer_text = answer_response.choices[0].message.content.strip()

    # ------------------ Step 7: Determine source ------------------
    if answer_text.startswith("[PDF-based answer]"):
        source_name = "Academy Answer"
        answer_text = answer_text.replace("[PDF-based answer]", "", 1).strip()
    elif answer_text.startswith("[GPT answer]"):
        source_name = "GPT Answer"
        answer_text = answer_text.replace("[GPT answer]", "", 1).strip()
    else:
        if top_chunks and any(word in answer_text.lower() for word in ["page", "pdf", "worksheet"]):
            source_name = "Academy Answer"
        else:
            source_name = "GPT Answer"

    # ------------------ Step 8: Append PDF metadata once ------------------
    if source_name == "Academy Answer" and top_chunks:
        top_doc = top_chunks[0]  # dict
        pdf_metadata = f"[PDF used: {top_doc.get('pdf_name','N/A')} (Page {top_doc.get('page_number','N/A')})]"
        answer_text = f"{answer_text}\n{pdf_metadata}"

    # ------------------ Step 9: Prepare final results ------------------
    used_pdfs = [c.get("pdf_link") for c in top_chunks if c.get("pdf_link")]

    results.append({
        "name": f"**{source_name}**",
        "snippet": answer_text,
        "links": used_pdfs if source_name == "Academy Answer" else []
    })

    # ------------------ Step 10: Update user context ------------------
    append_to_user_context(user_id, "user", query)
    append_to_user_context(user_id, "assistant", answer_text)

    print("==================== SEARCH REQUEST END ====================\n")
    return JSONResponse(results)




# Utility: hash password

# Bulk upload endpoint
def load_vectorstore_from_gcs_in_memory(gcs_prefix: str, embeddings: OpenAIEmbeddings) -> FAISS:
    """
    Load a FAISS vector store from GCS directly into memory.
    Handles docstore tuples or dict/InMemoryDocstore.
    """
    print(f"\n[DEBUG][GCS-LOAD] ======== Starting load_vectorstore_from_gcs_in_memory ========")
    print(f"[DEBUG][GCS-LOAD] Prefix: {gcs_prefix}")

    try:
        if not gcs_client or not gcs_bucket_name:
            raise RuntimeError("GCS client or bucket name not initialized")

        bucket = gcs_client.bucket(gcs_bucket_name)
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))
        if not blobs:
            raise ValueError(f"No vector store files found in GCS for prefix '{gcs_prefix}'")
        print(f"[DEBUG][GCS-LOAD] Found {len(blobs)} blobs in GCS prefix.")

        index_bytes, docstore_bytes = None, None
        for blob in blobs:
            if blob.name.endswith(".faiss"):
                index_bytes = blob.download_as_bytes()
                print(f"[DEBUG][GCS-LOAD] Found FAISS index file: {blob.name}")
            elif blob.name.endswith(".pkl"):
                docstore_bytes = blob.download_as_bytes()
                print(f"[DEBUG][GCS-LOAD] Found docstore file: {blob.name}")

        if not index_bytes or not docstore_bytes:
            raise RuntimeError(f"Missing FAISS index or docstore for prefix '{gcs_prefix}'")

        # Load FAISS index
        bio = io.BytesIO(index_bytes)
        reader = faiss.PyCallbackIOReader(bio.read)
        index = faiss.read_index(reader)
        print("[DEBUG][GCS-LOAD] FAISS index loaded into memory.")

        # Load docstore from pickle
        raw_docstore = pickle.loads(docstore_bytes)
        print(f"[DEBUG][GCS-LOAD] Raw docstore type: {type(raw_docstore)}")

        # Handle tuple (docstore, index_to_docstore_id) or dict/InMemoryDocstore
        if isinstance(raw_docstore, tuple) and len(raw_docstore) == 2:
            docstore, index_to_docstore_id = raw_docstore
            print("[DEBUG][GCS-LOAD] Unpacked tuple docstore and index_to_docstore_id")
        else:
            docstore = raw_docstore
            if hasattr(docstore, "_dict"):
                index_to_docstore_id = {i: doc_id for i, doc_id in enumerate(docstore._dict.keys())}
            elif isinstance(docstore, dict):
                index_to_docstore_id = {i: doc_id for i, doc_id in enumerate(docstore.keys())}
            else:
                raise RuntimeError(f"Unsupported docstore type: {type(docstore)}")
            print(f"[DEBUG][GCS-LOAD] index_to_docstore_id mapping created with {len(index_to_docstore_id)} items.")

        # Build inverse mapping
        index_to_docstore_id_inverse = {v: k for k, v in index_to_docstore_id.items()}

        # Construct LangChain FAISS
        vs = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        vs.index_to_docstore_id_inverse = index_to_docstore_id_inverse

        print(f"[DEBUG][GCS-LOAD] Successfully loaded FAISS store in memory.")
        print(f"[DEBUG][GCS-LOAD] ======== Finished load_vectorstore_from_gcs_in_memory ========\n")
        return vs

    except Exception as e:
        import traceback
        print(f"[ERROR][GCS-LOAD] Exception: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise


# ---------------------------
# Bulk upload embeddings endpoint
# ---------------------------
# ---------- Load FAISS vectorstore from GCS in memory ----------
def load_vectorstore_from_gcs_in_memory(gcs_prefix: str, embeddings: OpenAIEmbeddings) -> FAISS:
    """
    Load a FAISS vector store from GCS directly into memory, mimicking
    the successful behavior of load_vectorstore_from_gcs.
    """
    print(f"\n[DEBUG][GCS-LOAD] ======== Starting load_vectorstore_from_gcs_in_memory ========")
    print(f"[DEBUG][GCS-LOAD] Prefix: {gcs_prefix}")

    try:
        if not gcs_client or not gcs_bucket_name:
            raise RuntimeError("GCS client or bucket name not initialized")

        # List blobs under prefix
        bucket = gcs_client.bucket(gcs_bucket_name)
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))
        if not blobs:
            raise ValueError(f"No vector store files found in GCS for prefix '{gcs_prefix}'")
        print(f"[DEBUG][GCS-LOAD] Found {len(blobs)} blobs in GCS prefix.")

        # Find index and docstore files
        index_bytes = None
        docstore_bytes = None
        for blob in blobs:
            if blob.name.endswith(".faiss"):
                index_bytes = blob.download_as_bytes()
                print(f"[DEBUG][GCS-LOAD] Found FAISS index file: {blob.name}")
            elif blob.name.endswith(".pkl"):
                docstore_bytes = blob.download_as_bytes()
                print(f"[DEBUG][GCS-LOAD] Found docstore file: {blob.name}")

        if not index_bytes or not docstore_bytes:
            raise RuntimeError(f"Missing FAISS index or docstore for prefix '{gcs_prefix}'")

        # Load FAISS index in memory
        with io.BytesIO(index_bytes) as bio:
            reader = faiss.PyCallbackIOReader(bio.read)
            index = faiss.read_index(reader)
        print("[DEBUG][GCS-LOAD] FAISS index loaded into memory.")

        # Load docstore
        raw_docstore = pickle.loads(docstore_bytes)
        print(f"[DEBUG][GCS-LOAD] Raw docstore type: {type(raw_docstore)}")

        # Handle tuple (docstore, index_to_docstore_id) or dict/InMemoryDocstore
        if isinstance(raw_docstore, tuple) and len(raw_docstore) == 2:
            docstore, index_to_docstore_id = raw_docstore
            print("[DEBUG][GCS-LOAD] Unpacked tuple docstore and index_to_docstore_id")
        else:
            docstore = raw_docstore
            if hasattr(docstore, "_dict"):
                index_to_docstore_id = {i: doc_id for i, doc_id in enumerate(docstore._dict.keys())}
            elif isinstance(docstore, dict):
                index_to_docstore_id = {i: doc_id for i, doc_id in enumerate(docstore.keys())}
            else:
                raise RuntimeError(f"Unsupported docstore type: {type(docstore)}")
            print(f"[DEBUG][GCS-LOAD] index_to_docstore_id mapping created with {len(index_to_docstore_id)} items.")

        # Build inverse mapping
        index_to_docstore_id_inverse = {v: k for k, v in index_to_docstore_id.items()}

        # Construct FAISS vectorstore
        vs = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        vs.index_to_docstore_id_inverse = index_to_docstore_id_inverse

        print(f"[DEBUG][GCS-LOAD] Successfully loaded FAISS store in memory.")
        print(f"[DEBUG][GCS-LOAD] ======== Finished load_vectorstore_from_gcs_in_memory ========\n")
        return vs

    except Exception as e:
        import traceback
        print(f"[ERROR][GCS-LOAD] Exception: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise

@app.post("/admin/initialize_faiss")
def initialize_faiss(db: Session = Depends(get_db)):
    """
    Reads embeddings from the database and initializes a FAISS index in memory.
    """
    global FAISS_INDEX, FAISS_METADATA

    try:
        all_embeddings = db.query(Embedding).all()
        if not all_embeddings:
            raise HTTPException(status_code=404, detail="No embeddings found in DB.")

        # Convert embeddings to NumPy array
        vectors = np.array([e.embedding_vector for e in all_embeddings], dtype='float32')

        # Extract metadata for each embedding
        FAISS_METADATA = [
            {
                "pdf_name": e.pdf_name,
                "class_name": e.class_name,
                "page_number": e.page_number,
                "chunk_index": e.chunk_index,
                "pdf_link": e.pdf_link,
                "chunk_text": e.chunk_text
            }
            for e in all_embeddings
        ]

        # Build FAISS index
        d = vectors.shape[1]  # embedding dimension
        FAISS_INDEX = faiss.IndexFlatIP(d)
        faiss.normalize_L2(vectors)  # normalize for cosine similarity
        FAISS_INDEX.add(vectors)

        print(f"[DEBUG] FAISS index initialized with {len(all_embeddings)} embeddings.")

        return {"message": "FAISS index initialized", "num_embeddings": len(all_embeddings)}

    except Exception as e:
        print(f"[ERROR] Failed to initialize FAISS index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize FAISS index: {str(e)}")
     


@app.post("/admin/create_vectorstores")
async def create_vectorstores():
    print("[DEBUG] ===== Starting vector store processing =====")

    try:
        # Step 1: List all PDFs in the folder
        print(f"[DEBUG] Calling list_pdfs with DEMO_FOLDER_ID: {DEMO_FOLDER_ID}")
        all_pdfs = list_pdfs(DEMO_FOLDER_ID)
        print(f"[DEBUG] list_pdfs returned {len(all_pdfs)} PDFs:")
        for idx, pdf in enumerate(all_pdfs, start=1):
            print(f"    [{idx}] {pdf}")

        # Step 2: Ensure vector stores exist for all PDFs
        print("[DEBUG] Calling ensure_vectorstores_for_all_pdfs with all_pdfs")
        ensure_vectorstores_for_all_pdfs(all_pdfs)
        print("[DEBUG] ensure_vectorstores_for_all_pdfs completed successfully")

        print("[DEBUG] ===== Vector store processing finished successfully =====")
        return JSONResponse(
            status_code=200,
            content={"message": "Vector stores processed successfully!"}
        )

    except Exception as e:
        print(f"[ERROR] An exception occurred during vector store processing: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "An error occurred during vector store processing."}
        )


# ---------- Bulk upload embeddings endpoint ----------
# ---------- Bulk upload embeddings endpoint (updated) ----------
@app.post("/admin/upload_embeddings_to_db")
def upload_embeddings_to_db(db: Session = Depends(get_db)):
    """
    Loads FAISS vector stores for all PDFs from GCS in memory and inserts embeddings
    and chunk text into the DB, avoiding duplicates. Embeddings are stored as numeric arrays.
    """
    print("\n[DEBUG] Starting /admin/upload_embeddings_to_db endpoint")
    total_uploaded = 0
    total_skipped = 0

    try:
        # Step 0: Fetch PDFs
        print("[DEBUG] Fetching PDF list from Google Drive...")
        all_pdfs = list_pdfs(DEMO_FOLDER_ID)
        if not all_pdfs:
            raise HTTPException(status_code=404, detail="No PDFs found in Google Drive.")
        print(f"[DEBUG] Found {len(all_pdfs)} PDFs to process.")

        # Initialize embeddings model
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.environ.get("OPENAI_API_KEY_S")
        )

        # Step 1: Process vector stores
        for pdf_idx, pdf in enumerate(all_pdfs, start=1):
            pdf_name = pdf.get("name")
            pdf_path = pdf.get("path")
            pdf_base_name = pdf_name.rsplit(".", 1)[0]
            parent_folder = os.path.dirname(pdf_path)
            gcs_prefix = os.path.join(parent_folder, f"vectorstore_{pdf_base_name}") + "/"

            print(f"\n[DEBUG] Processing PDF {pdf_idx}/{len(all_pdfs)}: {pdf_name}")
            print(f"[DEBUG] Loading vector store from GCS: {gcs_prefix}")

            try:
                vs = load_vectorstore_from_gcs_in_memory(gcs_prefix, embeddings_model)
            except Exception as e:
                print(f"[WARNING] Could not load vector store for PDF {pdf_name}: {e}")
                continue

            # Step 2: Iterate over docstore items
            if hasattr(vs.docstore, "_dict"):
                docstore_items = vs.docstore._dict.items()
            elif isinstance(vs.docstore, dict):
                docstore_items = vs.docstore.items()
            else:
                raise RuntimeError(f"Unsupported docstore type: {type(vs.docstore)}")

            uploaded_this_pdf = 0
            for doc_id, doc in docstore_items:
                # Extract metadata and chunk content
                # Extract metadata and chunk content, sanitize null characters
                metadata = getattr(doc, "metadata", {}) or {}
                chunk_text = (metadata.get("chunk_text") or getattr(doc, "page_content", "") or "").replace("\x00", "")
                class_name = (metadata.get("class_name") or "").replace("\x00", "")
                pdf_link = (metadata.get("pdf_link") or "").replace("\x00", "")
                page_number = metadata.get("page_number", 0)
                chunk_index = metadata.get("chunk_index", 0)


                try:
                    # Reconstruct embedding as a numeric list
                    internal_id = vs.index_to_docstore_id_inverse[doc_id]
                    embedding_vector = vs.index.reconstruct(internal_id).tolist()
                except Exception as e:
                    print(f"[WARNING] Could not reconstruct embedding for doc_id {doc_id}: {e}")
                    continue

                # Skip duplicates
                existing = db.query(Embedding).filter_by(
                    pdf_name=pdf_name,
                    chunk_id=doc_id
                ).first()
                if existing:
                    total_skipped += 1
                    continue

                # Insert embedding into DB
                new_embedding = Embedding(
                    pdf_name=pdf_name,
                    class_name=class_name,
                    pdf_link=pdf_link,
                    chunk_text=chunk_text,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    chunk_id=doc_id,
                    embedding_vector=embedding_vector
                )
                db.add(new_embedding)
                total_uploaded += 1
                uploaded_this_pdf += 1

                if uploaded_this_pdf % 50 == 0:
                    print(f"[DEBUG] Uploaded {uploaded_this_pdf} embeddings for PDF '{pdf_name}' so far...")

            db.commit()
            print(f"[DEBUG] PDF '{pdf_name}': Uploaded {uploaded_this_pdf} embeddings. "
                  f"Total uploaded: {total_uploaded}, Total skipped: {total_skipped}")

        print(f"\n[DEBUG] Completed upload. Total uploaded: {total_uploaded}, Total skipped: {total_skipped}")
        return {
            "message": f"Uploaded {total_uploaded} new embeddings. Skipped {total_skipped} duplicates."
        }

    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to upload embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload embeddings: {str(e)}")








# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo_chatbot_backend_2:app", host="0.0.0.0", port=8000, reload=True)

