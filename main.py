import random
import time
from dotenv import load_dotenv
from typing import Optional, List, Dict
from typing import Literal
from zoneinfo import ZoneInfo
import pandas as pd
from cachetools import TTLCache
import re 
from langchain_core.documents import Document
import faiss
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email 

from pgvector.sqlalchemy import Vector
from fastapi.responses import HTMLResponse

import fitz  # PyMuPDF
import base64
from fastapi.responses import HTMLResponse  
from urllib.parse import urlencode


import numpy as np
#for protected pdf viewer for Gem Chatbot
PDF_CACHE = {}
PDF_CACHE_TTL_SECONDS = 60 * 10   # 10 minutes
ALLOWED_PDFS_CACHE = {}
ALLOWED_PDFS_CACHE_TTL_SECONDS = 60 * 5  



#Twilio API
from twilio.rest import Client
# FastAPI & Pydantic
from fastapi import FastAPI, Response, Depends, HTTPException, Query, Path, Body, UploadFile, File
from pydantic import BaseModel, EmailStr
from passlib.hash import pbkdf2_sha256

# SQLAlchemy
from sqlalchemy import or_, create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean, Text, text, Float, func, ARRAY, JSON, select, delete
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

from langchain_community.vectorstores import FAISS

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




AU_TZ = ZoneInfo("Australia/Sydney")

def australia_now():
    return datetime.now(AU_TZ).replace(tzinfo=None)


 

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

#for backend url so we can send proper backend pdf urls to chatbot users
BACKEND_PUBLIC_BASE_URL = os.getenv(
    "BACKEND_PUBLIC_BASE_URL",
    "http://127.0.0.1:8000"
)
# -----------------------------
# App & CORS
# -----------------------------
load_dotenv()
print("\n========== ENVIRONMENT CHECK ==========")

print("OPENAI_API_KEY_S:", bool(os.getenv("OPENAI_API_KEY")))
print("SENDGRID_API_KEY:", bool(os.getenv("SENDGRID_API_KEY")))
print("GCP_SERVICE_ACCOUNT_JSON:", bool(os.getenv("GCP_SERVICE_ACCOUNT_JSON")))
print("GOOGLE_APPLICATION_CREDENTIALS_JSON:", bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")))
print("DATABASE_URL:", bool(os.getenv("DATABASE_URL")))

print("OPENAI PREFIX:",
      os.getenv("OPENAI_API_KEY", "")[:12])

print("SENDGRID PREFIX:",
      os.getenv("SENDGRID_API_KEY", "")[:12])

print("=======================================\n")
app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://krish-chat-bot.vercel.app",
        "https://gamified-quiz-peach.vercel.app",
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
    api_key=os.environ.get("OPENAI_API_KEY")
)



FRONTEND_PUBLIC_BASE_URL = os.getenv(
    "FRONTEND_PUBLIC_BASE_URL",
    "http://localhost:3000"
).rstrip("/")


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

SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")

class GamifiedWelcomeQuoteResponse(BaseModel):
    quote: str
    author: str 
class ChatbotLoginSettings(Base):
    __tablename__ = "chatbot_login_settings"

    id = Column(Integer, primary_key=True, index=True)
    login_mode = Column(String, nullable=False, default="otp_only")

class ChatbotMessage(Base):
    __tablename__ = "chatbot_messages"

    id = Column(Integer, primary_key=True, index=True)

    conversation_id = Column(
        Integer,
        ForeignKey("chatbot_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # "user" or "assistant"
    role = Column(String, nullable=False)

    # actual message text
    message_text = Column(Text, nullable=False)

    # assistant metadata
    source_name = Column(String, nullable=True)       # Academy Answer / GPT Answer
    reasoning_level = Column(String, nullable=True)   # simple / medium / advanced
    class_name = Column(String, nullable=True)

    # optional PDF metadata if answer came from a booklet/pdf
    pdf_name = Column(String, nullable=True)
    pdf_page = Column(Integer, nullable=True)
    pdf_file_id = Column(String, nullable=True)

    # store returned links as JSON
    response_links = Column(JSON, nullable=True)

    created_at = Column(DateTime, nullable=False, default=australia_now)

    conversation = relationship("ChatbotConversation", back_populates="messages")


class ChatbotConversation(Base):
    __tablename__ = "chatbot_conversations"

    id = Column(Integer, primary_key=True, index=True)

    # one unique chat/session id
    conversation_uuid = Column(
        String,
        unique=True,
        nullable=False,
        index=True,
        default=lambda: str(uuid.uuid4())
    )

    # student / parent info copied at conversation start for easy admin filtering
    student_id = Column(
        String,
        nullable=False,
        index=True
    )

    student_name = Column(
        String,
        nullable=True
    )

    parent_email = Column(
        String,
        nullable=False,
        index=True
    )

    class_name = Column(
        String,
        nullable=True
    )

    student_year = Column(
        String,
        nullable=True
    )

    center_code = Column(
        String,
        nullable=True
    )

    center_name = Column(
        String,
        nullable=True
    )

    started_at = Column(
        DateTime,
        nullable=False,
        default=australia_now
    )

    last_message_at = Column(
        DateTime,
        nullable=False,
        default=australia_now
    )

    message_count = Column(
        Integer,
        nullable=False,
        default=0
    )

    status = Column(
        String,
        nullable=False,
        default="active"
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=australia_now
    )

    updated_at = Column(
        DateTime,
        nullable=False,
        default=australia_now
    )

    messages = relationship(
        "ChatbotMessage",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )


class ChatbotLoginSettingsResponse(BaseModel):
    login_mode: Literal["otp_only", "id_only", "both"]

    class Config:
        from_attributes = True


class ChatbotLoginSettingsUpdate(BaseModel):
    login_mode: Literal["otp_only", "id_only", "both"]

class AdminUser(Base):
    __tablename__ = "admin_users"

    id = Column(Integer, primary_key=True, index=True)

    username = Column(String, unique=True, nullable=False)

    password = Column(String, nullable=False)

    full_name = Column(String, nullable=False)

    role = Column(String, nullable=False)

    center_code = Column(String, nullable=True)

    email = Column(String, nullable=True)

    phone_number = Column(String, nullable=True)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

class Student(Base):

    __tablename__ = "students"

    id = Column(String, primary_key=True, index=True)

    # External student ID
    student_id = Column(
        String,
        unique=True,
        nullable=False
    )

    gender = Column(
        String,
        nullable=True
    )

    password = Column(
        String,
        nullable=False
    )

    name = Column(
        String,
        nullable=False
    )

    parent_email = Column(
        String,
        nullable=False
    )

    class_name = Column(
        String,
        nullable=False
    )

    class_day = Column(
        String,
        nullable=True
    )

    # NAPLAN year
    student_year = Column(
        String,
        nullable=False
    )

    # NEW
    center_code = Column(
        String,
        nullable=False
    )
    center_name = Column(
        String,
        nullable=False
    )


class KnowledgeBaseResponse(BaseModel):
    knowledge_base: Optional[str] = None

class CurrentTerm(Base):
    __tablename__ = "current_term"  # table name in DB

    id = Column(Integer, primary_key=True, autoincrement=True)
    term_name = Column(String(50), nullable=False)

class ChatbotCurrentTermRequest(BaseModel):
    term_name: str
 
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


class FranchiseLocation(Base):
    __tablename__ = "franchiselocation"
    id = Column(Integer, primary_key=True, autoincrement=True)
    country = Column(String(100), nullable=False)
    state = Column(String(100), nullable=False)

class UserListItem(BaseModel):
    id: int
    student_id: str               # <-- NEW FIELD
    name: str
    email: str
    phone_number: Optional[str] = None
    class_name: Optional[str] = None
    class_day: Optional[str] = None

    class Config:
        orm_mode = True  # allows SQLAlchemy models to be converted to Pydantic models

        

class UserResponse(BaseModel):
    name: str
    email: str
    phone_number: Optional[str] = None
    class_name: Optional[str] = None
    class_day: Optional[str] = None
    student_id: str                     # <-- NEW FIELD

    class Config:
        orm_mode = True
 

class UsageResponse(BaseModel):
    date: str
    amount_usd: float
    type: str

class SendOTPRequest(BaseModel):
    email: str


class VerifyOTPRequest(BaseModel):
    email: str
    otp: str


    
class AddUserRequest(BaseModel):
    name: str
    email: str
    phone_number: str
    class_name: str
    class_day: str
    student_id: str    # <-- REQUIRED
    password: str
 
class KnowledgeBaseRequest(BaseModel):
    knowledge_base: str

class EditUserRequest(BaseModel):
    name: str
    email: EmailStr
    phone_number: str
    class_name: str
    class_day: str
    student_id: str                 # <-- NEW FIELD
    password: Optional[str] = None  # only update if provided
 
class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RelevantWords(Base):
    __tablename__ = "relevant_words"
    id = Column(Integer, primary_key=True, autoincrement=True)
    singleton = Column(JSON)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    phone_number = Column(String, nullable=False)
    class_name = Column(String, nullable=False)
    class_day = Column(String, nullable=False)  # <-- added
    student_id = Column(String, unique=True, nullable=False)  # <-- NEW FIELD
    password = Column(String, nullable=False)
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

class LoginRequestChatbot(BaseModel):
    student_id: str
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
           class_day="N/A",  # <-- add this
           student_id="ADMIN001",
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
# -----------------------------
# Helper: get or create settings row
# -----------------------------
def get_or_create_chatbot_login_settings(db: Session):
    settings = db.query(ChatbotLoginSettings).first()

    if not settings:
        settings = ChatbotLoginSettings(login_mode="otp_only")
        db.add(settings)
        db.commit()
        db.refresh(settings)

    return settings


# -----------------------------
# GET chatbot login settings
# -----------------------------
@app.get("/chatbot-login-settings", response_model=ChatbotLoginSettingsResponse)
def get_chatbot_login_settings(db: Session = Depends(get_db)):
    settings = get_or_create_chatbot_login_settings(db)
    return settings


# -----------------------------
# PUT chatbot login settings
# -----------------------------
@app.put("/chatbot-login-settings", response_model=ChatbotLoginSettingsResponse)
def update_chatbot_login_settings(
    payload: ChatbotLoginSettingsUpdate,
    db: Session = Depends(get_db),
):
    settings = get_or_create_chatbot_login_settings(db)

    settings.login_mode = payload.login_mode

    db.commit()
    db.refresh(settings)

    return settings
@app.get("/chatbot-current-term")
def get_chatbot_current_term(
    db: Session = Depends(get_db)
):

    current_term = db.query(CurrentTerm).first()

    if not current_term:
        raise HTTPException(
            status_code=404,
            detail="Current chatbot term not configured."
        )

    return {
        "id": current_term.id,
        "term_name": current_term.term_name,
    }

@app.put("/chatbot-current-term")
def update_chatbot_current_term(
    data: ChatbotCurrentTermRequest,
    db: Session = Depends(get_db)
):

    current_term = db.query(CurrentTerm).first()

    if not current_term:

        current_term = CurrentTerm(
            term_name=data.term_name
        )

        db.add(current_term)

    else:

        current_term.term_name = data.term_name

    db.commit()
    db.refresh(current_term)

    return {
        "message": "Chatbot current term updated successfully.",
        "term": {
            "id": current_term.id,
            "term_name": current_term.term_name,
        }
    }

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
    return [
        UserListItem(
            id=u.id,
            student_id=u.student_id,   # <-- NEW FIELD
            name=u.name,
            email=u.email,
            phone_number=u.phone_number,
            class_name=u.class_name,
            class_day=u.class_day
        )
        for u in users
    ]
 
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
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
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
        


@app.get("/api/knowledge-base", response_model=KnowledgeBaseResponse)
def get_knowledge_base(db: Session = Depends(get_db)):
    kb_entry = db.query(KnowledgeBase).first()
    if not kb_entry:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    return KnowledgeBaseResponse(
        knowledge_base=kb_entry.content  # only this
    )
 
    
# ----------------- GET endpoint -----------------
# GET user by ID
@app.get("/users/info/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int = Path(..., description="ID of the user to retrieve"),
    db: Session = Depends(get_db)
):
    """
    Retrieve a user's information by ID for editing (excluding password).
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        name=user.name,
        email=user.email,
        phone_number=user.phone_number,
        class_name=user.class_name,
        class_day=user.class_day,
        student_id=user.student_id   # <-- NEW FIELD
    )
              
@app.get("/user_ids", response_model=List[UserListItem])
def get_user_ids(db: Session = Depends(get_db)):
    """
    Returns all users for the dropdown.
    """
    users = db.query(User).all()
    return users

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
        
"""   
sending otp using twilio sms services
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
"""

def send_otp_email(to_email: str, otp: str):
    # 🔎 TEMP DEBUG — remove after confirming
    print("[DEBUG] SENDGRID KEY PRESENT:", bool(SENDGRID_API_KEY))
    print("[DEBUG] SENDGRID KEY FORMAT OK:", SENDGRID_API_KEY.startswith("SG."))

    # Add these two lines here
    print("[DEBUG] KEY LENGTH:", len(SENDGRID_API_KEY))
    print("[DEBUG] KEY PREFIX:", SENDGRID_API_KEY[:15] + "...")

    message = Mail(
        from_email="noreply@gemkidsacademy.com.au",
        to_emails=to_email,
        subject="Your OTP Code",
        html_content=f"""
        <div style="
            max-width:600px;
            margin:0 auto;
            padding:40px 30px;
            font-family:Arial, Helvetica, sans-serif;
            color:#333333;
            background:#ffffff;
        ">

            <!-- Logo -->
            <div style="text-align:center; margin-bottom:30px;">
                <img
                    src="https://gemkidsacademy.com.au/wp-content/uploads/2024/10/cropped-logo-4-1.png"
                    alt="Gem Kids Academy"
                    style="width:180px;"
                />
            </div>

            <!-- Heading -->
            <h2 style="
                text-align:center;
                color:#2c3e50;
                margin-bottom:25px;
            ">
                OTP Code
            </h2>

            <p style="font-size:16px; line-height:1.6;">
                Your OTP code is below. Enter it in your open browser window to
                finish signing in.
            </p>

            <!-- OTP -->
            <div style="text-align:center; margin:35px 0;">
                <div style="
                    display:inline-block;
                    padding:18px 40px;
                    background:#f7f7f7;
                    border:2px solid rgb(0,140,200);
                    border-radius:10px;
                    font-size:34px;
                    font-weight:bold;
                    letter-spacing:8px;
                    color:rgb(219,71,45);
                ">
                    {otp}
                </div>
            </div>

            <p style="font-size:15px;">
                This code will expire in <strong>5 minutes</strong>.
            </p>

            <p style="font-size:15px; line-height:1.6;">
                If you didn't request this email, there's nothing to worry about—you
                can safely ignore it.
            </p>

            <p style="font-size:15px; line-height:1.6;">
                If you have any questions, please contact your Centre Manager or the
                <strong>Gem Kids Academy Support Team</strong>.
            </p>

            <hr style="
                margin:35px 0 20px 0;
                border:none;
                border-top:1px solid #e5e5e5;
            ">

            <p style="
                text-align:center;
                font-size:12px;
                color:#777777;
            ">
                © Gem Kids Academy
            </p>

        </div>
        """,
    )

    message.reply_to = Email('do-not-reply@gemkidsacademy.com.au')

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print(f"[INFO] Email sent to {to_email}, status code {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Failed to send email to {to_email}: {e}")
        raise

#sending otp using send grid 
@app.post("/send-otp")
def send_otp_endpoint(
    data: SendOTPRequest,
    db: Session = Depends(get_db)
):
    email = data.email.strip().lower()
    print(f"[DEBUG] Received OTP request for email: {email}")

    if not email:
        raise HTTPException(
            status_code=400,
            detail="Email is required"
        )

    # --------------------------------------------------
    # Check Admin Users first (case-insensitive)
    # --------------------------------------------------
    admin = (
        db.query(AdminUser)
        .filter(func.lower(AdminUser.email) == email)
        .first()
    )

    # --------------------------------------------------
    # If not found, check Students (Parent Email) case-insensitive
    # --------------------------------------------------
    student = None

    if not admin:
        student = (
            db.query(Student)
            .filter(func.lower(Student.parent_email) == email)
            .first()
        )

        if not student:
            print(f"[WARNING] Email {email} not found in AdminUser or Student tables")
            raise HTTPException(
                status_code=404,
                detail="Email not found"
            )

        print(f"[DEBUG] Found student '{student.name}' with parent email {student.parent_email}")

    else:
        print(f"[DEBUG] Found admin '{admin.full_name}' with email {admin.email}")

    # --------------------------------------------------
    # Generate OTP
    # --------------------------------------------------
    otp = generate_otp()
    print(f"[DEBUG] Generated OTP {otp} for email {email}")

    # --------------------------------------------------
    # Store OTP for 5 minutes
    # --------------------------------------------------
    otp_store[email] = {
        "otp": otp,
        "expiry": time.time() + 300
    }

    print(f"[DEBUG] Stored OTP for {email} with 5 minute expiry")

    # --------------------------------------------------
    # Send OTP Email
    # --------------------------------------------------
    try:
        print(f"[DEBUG] Sending OTP to {email}")
        send_otp_email(email, otp)
        print(f"[INFO] OTP successfully sent to {email}")

    except Exception as e:
        print(f"[ERROR] Failed to send OTP to {email}: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Error sending email: {str(e)}"
        )

    return {
        "message": "OTP sent successfully"
    }
def send_otp_endpoint(data: SendOTPRequest, db: Session = Depends(get_db)):
    email = data.email.strip().lower()
    print(f"[DEBUG] Received OTP request for email: {email}")

    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    # --- Lookup user by email ---
    user = db.query(User).filter(User.email == email).first()
    if not user:
        print(f"[WARNING] Email {email} not found in database")
        raise HTTPException(status_code=404, detail="Email not found")

    print(f"[DEBUG] Found user {user.name} with email {email}")

    # --- Generate OTP ---
    otp = generate_otp()
    print(f"[DEBUG] Generated OTP {otp} for email {email}")

    # --- Store OTP keyed by email with 5 min expiry ---
    otp_store[email] = {
        "otp": otp,
        "expiry": time.time() + 300
    }
    print(f"[DEBUG] Stored OTP for {email} with 5 min expiry")

    # --- Send OTP via email ---
    try:
        print(f"[DEBUG] Attempting to send OTP to {email} via email")
        send_otp_email(email, otp)
        print(f"[INFO] Successfully sent OTP to {email}")
    except Exception as e:
        print(f"[ERROR] Error sending OTP to {email}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error sending email: {str(e)}"
        )

    return {"message": "OTP sent successfully"}
"""
# not consistent with the new otp via email
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
"""
#consistent with the new otp via email
from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session
import time
from pydantic import BaseModel

# OTP store keyed by email
otp_store = {}

class VerifyOTPRequest(BaseModel):
    email: str
    otp: str


from fastapi import Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from sqlalchemy import func

@app.get("/admin/chatbot/conversations/{conversation_id}/messages")
def get_chatbot_conversation_messages(
    conversation_id: int,
    center_code: str,
    db: Session = Depends(get_db)
):
    # Ensure the conversation belongs to the logged-in centre
    conversation = (
        db.query(ChatbotConversation)
        .filter(
            ChatbotConversation.id == conversation_id,
            ChatbotConversation.center_code == center_code
        )
        .first()
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )

    messages = (
        db.query(ChatbotMessage)
        .filter(ChatbotMessage.conversation_id == conversation_id)
        .order_by(
            ChatbotMessage.created_at.asc(),
            ChatbotMessage.id.asc()
        )
        .all()
    )

    return {
        "conversation": {
            "id": conversation.id,
            "conversation_uuid": conversation.conversation_uuid,
            "student_id": conversation.student_id,
            "student_name": conversation.student_name,
            "parent_email": conversation.parent_email,
            "class_name": conversation.class_name,
            "student_year": conversation.student_year,
            "center_code": conversation.center_code,
            "center_name": conversation.center_name,
            "message_count": conversation.message_count,
            "status": conversation.status,
            "started_at": (
                conversation.started_at.isoformat()
                if conversation.started_at
                else None
            ),
            "last_message_at": (
                conversation.last_message_at.isoformat()
                if conversation.last_message_at
                else None
            ),
        },
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "message_text": msg.message_text,
                "source_name": msg.source_name,
                "reasoning_level": msg.reasoning_level,
                "class_name": msg.class_name,
                "pdf_name": msg.pdf_name,
                "pdf_page": msg.pdf_page,
                "pdf_file_id": msg.pdf_file_id,
                "response_links": msg.response_links or [],
                "created_at": (
                    msg.created_at.isoformat()
                    if msg.created_at
                    else None
                ),
            }
            for msg in messages
        ],
    }
@app.post("/welcome-quote", response_model=GamifiedWelcomeQuoteResponse)
def get_welcome_quote():
    print("\n==============================")
    print("WELCOME QUOTE(hi there)")
    print("==============================")

    fallback_quotes = [
        {
            "quote": "Success is the sum of small efforts, repeated day in and day out.",
            "author": "Robert Collier",
        },
        {
            "quote": "Learning never exhausts the mind.",
            "author": "Leonardo da Vinci",
        },
        {
            "quote": "The beautiful thing about learning is that no one can take it away from you.",
            "author": "B.B. King",
        },
        {
            "quote": "Education is the passport to the future, for tomorrow belongs to those who prepare for it today.",
            "author": "Malcolm X",
        },
        {
            "quote": "The expert in anything was once a beginner.",
            "author": "Helen Hayes",
        },
    ]

    try:
        prompt = """
You are generating a welcome quote for a school student.

Return exactly ONE short educational or inspirational quote suitable for students.

Rules:
- The quote must have a real author.
- Do not invent authors.
- Do not include any explanation.

Return ONLY valid JSON in this exact format:

{
  "quote": "string",
  "author": "string"
}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You generate short motivational educational quotes for students in strict JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=0.8,
        )

        raw_content = response.choices[0].message.content.strip()

        print("Raw GPT quote response:", raw_content)

        parsed = json.loads(raw_content)

        quote = (parsed.get("quote") or "").strip()
        author = (parsed.get("author") or "").strip()

        if not quote or not author:
            raise ValueError("Quote or author missing in GPT response")

        return GamifiedWelcomeQuoteResponse(
            quote=quote,
            author=author,
        )

    except Exception as e:
        print("Error generating welcome quote:", str(e))

        fallback = random.choice(fallback_quotes)

        return GamifiedWelcomeQuoteResponse(
            quote=fallback["quote"],
            author=fallback["author"],
        )
@app.get("/admin/chatbot/conversation-students")
def get_chatbot_conversation_students(
    center_code: str,
    date: str = Query(...),   # YYYY-MM-DD
    db: Session = Depends(get_db)
):
    rows = (
        db.query(
            ChatbotConversation.student_id,
            ChatbotConversation.student_name
        )
        .filter(
            ChatbotConversation.center_code == center_code,
            func.date(ChatbotConversation.started_at) == date
        )
        .distinct()
        .order_by(ChatbotConversation.student_id.asc())
        .all()
    )

    return [
        {
            "student_id": row.student_id,
            "student_name": row.student_name,
        }
        for row in rows
    ]

@app.get("/admin/chatbot/conversations")
def get_chatbot_conversations(
    center_code: str,
    date: str = Query(None),   # expected format: YYYY-MM-DD
    student_id: str = Query(None),
    db: Session = Depends(get_db)
):
    # Always restrict data to the logged-in user's centre
    query = (
        db.query(ChatbotConversation)
        .filter(ChatbotConversation.center_code == center_code)
    )

    # Optional date filter
    if date:
        query = query.filter(
            func.date(ChatbotConversation.started_at) == date
        )

    # Optional student filter
    if student_id:
        query = query.filter(
            ChatbotConversation.student_id == student_id
        )

    conversations = (
        query.order_by(desc(ChatbotConversation.last_message_at))
        .all()
    )

    return [
        {
            "id": convo.id,
            "conversation_uuid": convo.conversation_uuid,
            "student_id": convo.student_id,
            "student_name": convo.student_name,
            "parent_email": convo.parent_email,
            "class_name": convo.class_name,
            "student_year": convo.student_year,
            "center_code": convo.center_code,
            "center_name": convo.center_name,
            "message_count": convo.message_count,
            "status": convo.status,
            "started_at": (
                convo.started_at.isoformat()
                if convo.started_at
                else None
            ),
            "last_message_at": (
                convo.last_message_at.isoformat()
                if convo.last_message_at
                else None
            ),
        }
        for convo in conversations
    ]
@app.post("/verify-otp")
def verify_otp(
    data: VerifyOTPRequest,
    db: Session = Depends(get_db)
):
    email = data.email.strip().lower()
    print(f"[DEBUG] Received OTP verification request for email: {email}")

    # --------------------------------------------------
    # Check AdminUser first
    # --------------------------------------------------
    admin = (
        db.query(AdminUser)
        .filter(AdminUser.email == email)
        .first()
    )

    student = None

    # --------------------------------------------------
    # If not an admin, check Student table
    # --------------------------------------------------
    if not admin:
        student = (
            db.query(Student)
            .filter(Student.parent_email == email)
            .first()
        )

        if not student:
            print(f"[WARNING] Email {email} not found in AdminUser or Student tables")
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

    # --------------------------------------------------
    # Retrieve OTP record
    # --------------------------------------------------
    record = otp_store.get(email)

    if not record:
        print(f"[WARNING] No OTP record found for {email}")
        raise HTTPException(
            status_code=400,
            detail="OTP not sent"
        )

    if time.time() > record["expiry"]:
        print(f"[WARNING] OTP for {email} has expired")
        otp_store.pop(email, None)

        raise HTTPException(
            status_code=400,
            detail="OTP expired"
        )

    if str(data.otp) != str(record["otp"]):
        print(
            f"[WARNING] Entered OTP ({data.otp}) does not match stored OTP ({record['otp']})"
        )

        raise HTTPException(
            status_code=400,
            detail="Invalid OTP"
        )

    print(f"[INFO] OTP for {email} is valid")

    # Remove OTP after successful verification
    otp_store.pop(email, None)

    # --------------------------------------------------
    # Admin Login
    # --------------------------------------------------
    if admin:

        print(f"[INFO] Admin '{admin.full_name}' authenticated")

        if admin.full_name in user_contexts:
            user_contexts[admin.full_name] = []
            print(f"[DEBUG] Cleared previous context for admin {admin.full_name}")

        user_vectorstores_initialized[admin.full_name] = False

        admin_info = {
            "id": admin.id,
            "name": admin.full_name,
            "username": admin.username,
            "email": admin.email,
            "role": admin.role,
            "center_code": admin.center_code,
        }

        return {
            "message": "OTP verified successfully",
            "user": admin_info,
        }

    # --------------------------------------------------
    # Student Login
    # --------------------------------------------------
    print(f"[INFO] Student '{student.name}' authenticated")

    if student.name in user_contexts:
        user_contexts[student.name] = []
        print(f"[DEBUG] Cleared previous context for student {student.name}")

    user_vectorstores_initialized[student.name] = False

    student_info = {
        "id": student.id,
        "student_id": student.student_id,
        "name": student.name,
        "email": student.parent_email,
        "class_name": student.class_name,
        "class_day": student.class_day,
        "student_year": student.student_year,
        "center_code": student.center_code,
        "center_name": student.center_name,
    }

    return {
        "message": "OTP verified successfully",
        "user": student_info,
    }
@app.post("/login-GemKidsAcademyChatbot")
def login(
    data: LoginRequestChatbot,
    db: Session = Depends(get_db)
):
    student_id = data.student_id.strip()
    password = data.password.strip()

    print(f"[DEBUG] Login request received for ID: {student_id}")

    if not student_id or not password:
        raise HTTPException(
            status_code=400,
            detail="Student ID and password are required"
        )

    # --------------------------------------------------
    # Check Admin Users first
    # --------------------------------------------------
    admin = (
        db.query(AdminUser)
        .filter(AdminUser.username == student_id)
        .first()
    )

    if admin:

        if admin.password != password:
            print(f"[WARNING] Invalid password for admin {student_id}")
            raise HTTPException(
                status_code=401,
                detail="Invalid ID or password"
            )

        print(f"[INFO] Admin '{admin.full_name}' logged in")

        if admin.full_name in user_contexts:
            user_contexts[admin.full_name] = []

        user_vectorstores_initialized[admin.full_name] = False

        admin_info = {
            "id": admin.id,
            "name": admin.full_name,
            "username": admin.username,
            "role": admin.role,
            "email": admin.email,
            "center_code": admin.center_code,
        }

        return {
            "message": "Login successful",
            "user": admin_info,
        }

    # --------------------------------------------------
    # Check Students
    # --------------------------------------------------
    student = (
        db.query(Student)
        .filter(Student.student_id == student_id)
        .first()
    )

    if not student:
        print(f"[WARNING] Student ID '{student_id}' not found")

        raise HTTPException(
            status_code=401,
            detail="Invalid ID or password"
        )

    if student.password != password:
        print(f"[WARNING] Invalid password for student {student_id}")

        raise HTTPException(
            status_code=401,
            detail="Invalid ID or password"
        )

    print(f"[INFO] Student '{student.name}' logged in")

    if student.name in user_contexts:
        user_contexts[student.name] = []

    user_vectorstores_initialized[student.name] = False

    student_info = {
        "id": student.id,
        "student_id": student.student_id,
        "name": student.name,
        "email": student.parent_email,
        "class_name": student.class_name,
        "class_day": student.class_day,
        "student_year": student.student_year,
        "center_code": student.center_code,
        "center_name": student.center_name,
    }

    return {
        "message": "Login successful",
        "user": student_info,
    }

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

"""
def give_drive_access(file_id: str, emails: str, role: str = "reader", db: Session = None):
    
    #Grants Google Drive access to the folder(s) matching the user's class name(s) from the database.

    #:param file_id: Root Drive folder ID (DEMO_FOLDER_ID)
    #:param emails: Comma-separated string of user emails
    #:param role: "reader" or "writer"
    #:param db: SQLAlchemy Session (must be provided)
    
    if db is None:
        raise ValueError("A database session must be provided via `db` argument.")

    print("==== Starting Drive access process ====")

    # Split and clean the email list
    email_list = [email.strip() for email in emails.split(",") if email.strip()]
    if not email_list:
        print("No emails provided. Exiting.")
        return

    print(f"DEBUG: Emails to process: {email_list}")

    # Fetch users from DB
    users = db.query(User).filter(User.email.in_(email_list)).all()
    if not users:
        print("No matching users found in the database. Exiting.")
        return

    # Warn if any emails not found in DB
    missing_emails = set(email_list) - {u.email for u in users}
    if missing_emails:
        print(f"WARNING: These emails were not found in DB: {missing_emails}")

    # Fetch all folders under root DEMO_FOLDER_ID
    try:
        response = drive_service.files().list(
            q=f"'{file_id}' in parents and mimeType='application/vnd.google-apps.folder'",
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        folders = response.get("files", [])
        print(f"DEBUG: Found folders in Drive: {[f['name'] for f in folders]}")
    except HttpError as e:
        print(f"ERROR: Failed to fetch folders from Drive: {e}")
        return

    # Map folder names to IDs for faster lookup
    folder_map = {folder["name"].strip().lower(): folder["id"] for folder in folders}

    # Grant access to each user based on their class_name(s)
    for user in users:
        if not user.class_name:
            print(f"Skipping {user.email}: no class_name in DB")
            continue

        # Handle multiple class names, e.g., "Year 1, Year 2"
        user_classes = [cn.strip().lower() for cn in user.class_name.split(",") if cn.strip()]
        print(f"DEBUG: Processing {user.email} with class(es): {user_classes}")

        for cls in user_classes:
            folder_id_to_share = folder_map.get(cls)
            if not folder_id_to_share:
                print(f"WARNING: No folder found for class '{cls}', skipping {user.email}")
                continue

            print(f"DEBUG: Attempting to share folder '{cls}' (ID: {folder_id_to_share}) with {user.email}")
            try:
                drive_service.permissions().create(
                    fileId=folder_id_to_share,
                    body={
                        "type": "user",
                        "role": role,
                        "emailAddress": user.email
                    },
                    fields="id",
                    sendNotificationEmail=False
                ).execute()
                print(f"DEBUG: Drive API call succeeded for {user.email} on folder '{cls}'")
            except HttpError as error:
                print(f"ERROR: Drive API call failed for {user.email} on folder '{cls}': {error}")

    print("==== Drive access process completed ====")
"""

from googleapiclient.errors import HttpError


from googleapiclient.errors import HttpError


def log_drive_folder_hierarchy(root_folder_id: str):
    """
    Logs Google Drive hierarchy:

    Root
      └── Class
            └── Year
                  └── Term
                        └── PDF files
    """

    print("\n================ DRIVE FOLDER HIERARCHY ================\n")

    def get_subfolders(parent_id: str):
        try:
            response = drive_service.files().list(
                q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                spaces="drive",
                fields="files(id,name)",
                orderBy="name"
            ).execute()

            return response.get("files", [])

        except HttpError as e:
            print(f"ERROR fetching subfolders for parent {parent_id}: {e}")
            return []

    def get_files(parent_id: str):
        """
        Returns all non-folder files inside a Google Drive folder.
        Handles pagination and prints everything it finds.
        """

        files = []
        page_token = None

        while True:

            response = drive_service.files().list(
                q=(
                    f"'{parent_id}' in parents "
                    "and mimeType != 'application/vnd.google-apps.folder' "
                    "and trashed = false"
                ),
                spaces="drive",
                fields="nextPageToken, files(id,name,mimeType,size)",
                orderBy="name",
                pageToken=page_token,
            ).execute()

            current_files = response.get("files", [])

            files.extend(current_files)

            page_token = response.get("nextPageToken")

            if page_token is None:
                break

        print(f"\nFolder {parent_id}")
        print(f"Found {len(files)} file(s)")

        if not files:
            print("    (No files found)")
        else:
            for file in files:

                print(f"    Name : {file['name']}")
                print(f"    ID   : {file['id']}")
                print(f"    MIME : {file['mimeType']}")

                if "size" in file:
                    print(f"    Size : {file['size']} bytes")

                print("-" * 60)

        return files
    try:

        root_meta = drive_service.files().get(
            fileId=root_folder_id,
            fields="id,name"
        ).execute()

        print(f"ROOT: {root_meta['name']} ({root_meta['id']})\n")

        class_folders = get_subfolders(root_folder_id)

        if not class_folders:
            print("No class folders found.")
            return

        for class_folder in class_folders:

            print(
                f"├── CLASS: {class_folder['name']} ({class_folder['id']})"
            )

            year_folders = get_subfolders(class_folder["id"])

            if not year_folders:
                print("│   └── No year folders")
                continue

            for year_folder in year_folders:

                print(
                    f"│   ├── YEAR: {year_folder['name']} ({year_folder['id']})"
                )

                term_folders = get_subfolders(year_folder["id"])

                if not term_folders:
                    print("│   │   └── No term folders")
                    continue

                for term_folder in term_folders:

                    print(
                        f"│   │   ├── TERM: {term_folder['name']} ({term_folder['id']})"
                    )
                    
                    files = get_files(term_folder["id"])

                    if not files:
                        print("│   │   │   └── (No files)")
                        continue

                    for file in files:

                        icon = "📄"

                        if file["mimeType"] == "application/pdf":
                            icon = "📕"

                        print(
                            f"│   │   │   ├── {icon} "
                            f"{file['name']} "
                            f"({file['id']})"
                        )

        print("\n================ END OF DRIVE HIERARCHY ================\n")

    except HttpError as e:

        print("ERROR while reading Drive hierarchy")
        print(str(e))

log_drive_folder_hierarchy(DEMO_FOLDER_ID)

def give_drive_access(file_id: str, emails: str, role: str = "reader", db: Session = None):
    """
    Grants Google Drive access to a specific Year/Term folder:
      Root Folder → Year Folder → Term Folder

    :param file_id: Root Drive folder ID (e.g., GEM_AI_ROOT_FOLDER)
    :param emails: Comma-separated list of emails
    :param role: Drive permission role ('reader' or 'writer')
    :param db: SQLAlchemy Session
    """

    if db is None:
        raise ValueError("A database session must be provided via `db` argument.")

    print("==== Starting Drive access process ====")

    # -------------------------
    # Split & sanitize emails
    # -------------------------
    email_list = [email.strip() for email in emails.split(",") if email.strip()]
    if not email_list:
        print("No emails provided. Exiting.")
        return

    print(f"DEBUG: Emails to process: {email_list}")


    # -------------------------
    # Fetch students using parent_email
    # -------------------------
    students = db.query(Student).filter(Student.parent_email.in_(email_list)).all()

    if not students:
        print("No matching students found in DB. Exiting.")
        return

    found_parent_emails = {s.parent_email.strip().lower() for s in students if s.parent_email}
    missing = {email.strip().lower() for email in email_list} - found_parent_emails

    if missing:
        print(f"WARNING: Parent emails not found in DB: {missing}")

    print("DEBUG: Students fetched from DB:")
    for s in students:
        print(
            f"    student_id={s.student_id}, "
            f"name={s.name}, "
            f"parent_email={s.parent_email}, "
            f"class_name={s.class_name}, "
            f"student_year={s.student_year}"
        )

    # -------------------------
    # Load current term from DB
    # -------------------------
    current_term_record = db.query(CurrentTerm).first()
    if not current_term_record:
        print("ERROR: No current term found in DB. Exiting.")
        return

    current_term = current_term_record.term_name.strip().lower()
    print(f"DEBUG: Current term from DB: '{current_term}'")

    # -------------------------
    # Helper to fetch subfolders
    # -------------------------
    def get_subfolders(parent_id: str):
        try:
            response = drive_service.files().list(
                q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder'",
                spaces="drive",
                fields="files(id, name)"
            ).execute()
            return response.get("files", [])
        except HttpError as e:
            print(f"ERROR: Failed fetching subfolders for parent {parent_id}: {e}")
            return []

    # -------------------------
    # 1. Fetch Class folders under root
    # -------------------------
    class_folders = get_subfolders(file_id)
    print(f"DEBUG: Found Class folders: {[f['name'] for f in class_folders]}")

    # -------------------------
    # 2. Build Class → Year → Term → folderId map
    # -------------------------
    folder_map = {}

    for class_folder in class_folders:
        class_name = class_folder["name"].strip().lower()
        class_id = class_folder["id"]

        year_folders = get_subfolders(class_id)

        folder_map[class_name] = {}

        for year_folder in year_folders:
            year_name = year_folder["name"].strip().lower()
            year_id = year_folder["id"]

            term_folders = get_subfolders(year_id)

            folder_map[class_name][year_name] = {
                term["name"].strip().lower(): term["id"]
                for term in term_folders
            }

    print("DEBUG: Folder map:")
    for class_name, years in folder_map.items():
        print(f"  CLASS: {class_name}")
        for year_name, terms in years.items():
            print(f"    YEAR: {year_name}")
            print(f"      TERMS: {list(terms.keys())}")

    # -------------------------
    # 3. Grant access
    # -------------------------
    for student in students:

        if not student.class_name:
            print(f"Skipping {student.parent_email}: class_name is empty.")
            continue

        print(f"DEBUG: Raw class_name from DB = '{student.class_name}'")
        print(f"DEBUG: Raw student_year from DB = '{student.student_year}'")

        student_classes = [c.strip().lower() for c in student.class_name.split(",") if c.strip()]
        student_year = student.student_year.strip().lower() if student.student_year else ""

        print(
            f"DEBUG: Processing student_id={student.student_id}, "
            f"parent_email={student.parent_email}, "
            f"class_name={student_classes}, "
            f"student_year={student_year}"
        )

        if not student_year:
            print(f"Skipping {student.parent_email}: student_year is empty.")
            continue

        for class_key in student_classes:

            # Check Class folder exists
            if class_key not in folder_map:
                print(
                    f"WARNING: Class folder '{class_key}' not found in Drive. "
                    f"Available class folders: {list(folder_map.keys())}"
                )
                continue

            # Check Year folder exists under class
            if student_year not in folder_map[class_key]:
                print(
                    f"WARNING: Year folder '{student_year}' not found under Class '{class_key}'. "
                    f"Available years: {list(folder_map[class_key].keys())}"
                )
                continue

            # Check Term folder exists under class/year
            if current_term not in folder_map[class_key][student_year]:
                print(
                    f"WARNING: Term '{current_term}' not found under "
                    f"Class '{class_key}' → Year '{student_year}'."
                )
                continue

            # Final target folder
            folder_id_to_share = folder_map[class_key][student_year][current_term]

            print(f"DRIVE URL: https://drive.google.com/drive/folders/{folder_id_to_share}")
            print(
                f"DEBUG: Sharing Class='{class_key}', Year='{student_year}', "
                f"Term='{current_term}' (ID={folder_id_to_share}) "
                f"with parent {student.parent_email}"
            )

            try:
                print("--------------------------------------------------")
                print(f"Creating permission...")
                print(f"Parent Email : {student.parent_email}")
                print(f"Folder ID  : {folder_id_to_share}")
                print(f"Role       : {role}")
                print("--------------------------------------------------")
                folder_meta = drive_service.files().get(
                    fileId=folder_id_to_share,
                    fields="id,name"
                ).execute()

                print("DEBUG: Target Folder Metadata")
                print(folder_meta)

                permission = drive_service.permissions().create(
                    fileId=folder_id_to_share,
                    body={
                        "type": "user",
                        "role": role,
                        "emailAddress": student.parent_email
                    },
                    fields="id",
                    sendNotificationEmail=False
                ).execute()
                print(
                    f"SUCCESS: Permission created. "
                    f"Permission ID={permission.get('id')}"
                )

                # Verify permission exists
                permissions = drive_service.permissions().list(
                    fileId=folder_id_to_share,
                    fields="permissions(id,emailAddress,role)"
                ).execute()

                print("DEBUG: Folder permissions after share:")

                user_found = False

                for p in permissions.get("permissions", []):
                    print(
                        f"    email={p.get('emailAddress')} "
                        f"role={p.get('role')} "
                        f"id={p.get('id')}"
                    )

                    if p.get("emailAddress", "").lower() == student.parent_email.lower():
                        user_found = True

                if user_found:
                    print(f"VERIFIED: {student.parent_email} exists in folder permissions")
                else:
                    print(f"WARNING: {student.parent_email} NOT FOUND in folder permissions")

                print(
                    f"SUCCESS: Permission created. "
                    f"Permission ID={permission.get('id')}"
                )

                print(f"SUCCESS: Shared folder {folder_id_to_share} with {student.parent_email}")

            except HttpError as e:

                print("========== GOOGLE DRIVE ERROR ==========")
                print(f"Parent Email : {student.parent_email}")
                print(f"Folder ID  : {folder_id_to_share}")

                try:
                    print("Response:")
                    print(e.content.decode())
                except:
                    pass

                print(str(e))
                print("========================================")

    print("==== Drive access process completed ====")




@app.get("/api/knowledge-base", response_model=KnowledgeBaseResponse)
def get_knowledge_base(db: Session = Depends(get_db)):
    kb_entry = db.query(KnowledgeBase).first()
    if not kb_entry:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    return KnowledgeBaseResponse(
        knowledge_base=kb_entry.content  # <-- use 'content', not 'knowledge_base'
    )
#here
@app.post("/add_user")
def add_user(user_request: AddUserRequest, db: Session = Depends(get_db)):

    print("\n========== ADD USER REQUEST ==========")
    print(f"Name        : {user_request.name}")
    print(f"Email       : {user_request.email}")
    print(f"Student ID  : {user_request.student_id}")
    print(f"Class Name  : {user_request.class_name}")
    print(f"Class Day   : {user_request.class_day}")

    # ---------- Check for duplicate email ----------
    existing_email = db.query(User).filter(
        User.email == user_request.email
    ).first()

    if existing_email:
        print(f"ERROR: Email already exists -> {user_request.email}")
        raise HTTPException(status_code=400, detail="Email already registered")

    # ---------- Check for duplicate student_id ----------
    existing_student = db.query(User).filter(
        User.student_id == user_request.student_id
    ).first()

    if existing_student:
        print(f"ERROR: Student ID already exists -> {user_request.student_id}")
        raise HTTPException(status_code=400, detail="Student ID already registered")

    print("DEBUG: Generating password hash")

    hashed_password = generate_password_hash(user_request.password)

    print("DEBUG: Creating user record")

    new_user = User(
        name=user_request.name,
        email=user_request.email,
        phone_number=user_request.phone_number,
        class_name=user_request.class_name,
        class_day=user_request.class_day,
        student_id=user_request.student_id,
        password=hashed_password,
        created_at=datetime.utcnow()
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    print(f"SUCCESS: User created with DB ID={new_user.id}")

    print(
        f"DEBUG: Calling give_drive_access() "
        f"for email={user_request.email}"
    )

    try:
        give_drive_access(
            DEMO_FOLDER_ID,
            user_request.email,
            role="reader",
            db=db
        )
    except Exception as e:
        print("ERROR: give_drive_access failed")
        print(str(e))
        raise

    print("========== ADD USER COMPLETE ==========\n")

    return {
        "message": (
            f"User '{new_user.name}' added successfully "
            f"with Student ID '{new_user.student_id}'. "
            f"Drive access granted!"
        )
    }

@app.put("/edit-user/{user_id}")
def edit_user(
    user_id: int = Path(..., description="ID of the user to update"),
    user_request: EditUserRequest = Body(...),
    db: Session = Depends(get_db)
):
    # ----- Fetch the user -----
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # ----- Check if email is being updated -----
    if user.email != user_request.email:
        existing_email = db.query(User).filter(User.email == user_request.email).first()
        if existing_email:
            raise HTTPException(status_code=400, detail="Email already registered")

    # ----- Check if student_id is being updated -----
    if user.student_id != user_request.student_id:
        existing_student = db.query(User).filter(
            User.student_id == user_request.student_id
        ).first()
        if existing_student:
            raise HTTPException(status_code=400, detail="Student ID already registered")

    # ----- Apply updates -----
    user.name = user_request.name
    user.email = user_request.email
    user.phone_number = user_request.phone_number
    user.class_name = user_request.class_name
    user.class_day = user_request.class_day
    user.student_id = user_request.student_id  # <-- NEW FIELD UPDATE

    # ----- Update password only if provided -----
    if user_request.password:
        user.password = generate_password_hash(user_request.password)

    user.updated_at = datetime.utcnow()  # optional tracking

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
            openai_api_key=os.environ.get("OPENAI_API_KEY")
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

def fix_missing_pdf_links(db: Session = Depends(get_db)):
    """
    Finds embeddings with missing pdf_link, fetches correct links from Google Drive,
    and updates DB. Does not touch embeddings themselves.
    """
    try:
        # -------------------- Step 1: Find embeddings with missing pdf_link --------------------
        missing_links = db.query(Embedding).filter(
            (Embedding.pdf_link == None) | (Embedding.pdf_link == "")
        ).all()
        print(f"[INFO] Found {len(missing_links)} embeddings with missing pdf_link")
        if not missing_links:
            return {"message": "No missing pdf_link found. All good!"}

        # -------------------- Step 2: Fetch correct pdf_link from Google Drive --------------------
        all_pdfs = list_pdfs(DEMO_FOLDER_ID)  # returns list of {name, path, webViewLink, ...}
        pdf_link_map = {pdf["name"]: pdf.get("webViewLink", "") for pdf in all_pdfs}

        # -------------------- Step 3: Update embeddings --------------------
        updated_count = 0
        for emb in missing_links:
            correct_link = pdf_link_map.get(emb.pdf_name)
            if correct_link:
                emb.pdf_link = correct_link
                updated_count += 1

        db.commit()
        print(f"[INFO] Updated {updated_count} embeddings with missing pdf_link")

        return {
            "message": f"Updated {updated_count} embeddings with correct pdf_link",
            "total_missing_before": len(missing_links)
        }

    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to fix missing pdf_link: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fix missing pdf_link: {str(e)}")




processed_pdfs = set()  # Keep track of PDFs processed in this session

def ensure_vectorstores_for_all_pdfs(pdf_files):
    """
    Diagnostic version:
    Logs whether we are checking the PDF blob instead of the vectorstore prefix.
    Does NOT change the behavior yet — it only makes the current decision visible.
    """
    print("\n================ ensure_vectorstores_for_all_pdfs START ================\n")

    for idx, pdf in enumerate(pdf_files, start=1):
        print(f"\n---------------- PDF #{idx} ----------------")

        pdf_id = pdf.get("id")
        pdf_name = pdf.get("name", "Unknown")
        pdf_path = pdf.get("path", "")

        print(f"[DEBUG] pdf_id   = {pdf_id}")
        print(f"[DEBUG] pdf_name = {pdf_name}")
        print(f"[DEBUG] pdf_path = {pdf_path}")

        if not pdf_id:
            print(f"[DEBUG] PDF {pdf_name} has no Drive ID. Skipping.")
            continue

        # --------------------------------------------------
        # Build the vectorstore prefix we EXPECT for this PDF
        # --------------------------------------------------
        parent_folder = pdf_path.rsplit("/", 1)[0] if "/" in pdf_path else ""
        pdf_base_name = pdf_name.rsplit(".", 1)[0]
        expected_vectorstore_prefix = f"{parent_folder}/vectorstore_{pdf_base_name}/".replace("\\", "/")

        print(f"[DEBUG] parent_folder               = {parent_folder}")
        print(f"[DEBUG] pdf_base_name               = {pdf_base_name}")
        print(f"[DEBUG] expected_vectorstore_prefix = {expected_vectorstore_prefix}")

        # --------------------------------------------------
        # Check session memory
        # --------------------------------------------------
        if pdf_id in processed_pdfs:
            print(f"[DEBUG] PDF {pdf_name} already processed in this session. Skipping.")
            continue

        # --------------------------------------------------
        # Check current logic: does blob(pdf_path) exist?
        # --------------------------------------------------
        try:
            pdf_blob_exists = gcs_bucket.blob(pdf_path).exists()
        except Exception as e:
            pdf_blob_exists = f"ERROR: {e}"

        print(f"[DEBUG] gcs_bucket.blob(pdf_path).exists() = {pdf_blob_exists}")

        # --------------------------------------------------
        # Check what we ACTUALLY care about:
        # are there any files under the expected vectorstore prefix?
        # --------------------------------------------------
        try:
            vectorstore_blobs = list(
                gcs_bucket.list_blobs(prefix=expected_vectorstore_prefix, max_results=10)
            )
            vectorstore_blob_names = [b.name for b in vectorstore_blobs]
        except Exception as e:
            vectorstore_blobs = []
            vectorstore_blob_names = [f"ERROR: {e}"]

        print(f"[DEBUG] vectorstore blob count under expected prefix = {len(vectorstore_blobs)}")
        if vectorstore_blob_names:
            print("[DEBUG] vectorstore blobs found:")
            for name in vectorstore_blob_names:
                print(f"        - {name}")
        else:
            print("[DEBUG] No vectorstore blobs found under expected prefix.")

        # --------------------------------------------------
        # Show the current decision path without changing it
        # --------------------------------------------------
        if pdf_blob_exists:
            print(
                f"[DEBUG] CURRENT LOGIC DECISION: "
                f"Skip {pdf_name} because gcs_bucket.blob(pdf_path).exists() returned True."
            )

            if len(vectorstore_blobs) == 0:
                print(
                    f"[WARNING] Suspicious case: PDF blob exists, but NO vectorstore blobs were found "
                    f"under {expected_vectorstore_prefix}. "
                    f"This suggests the current skip condition may be wrong."
                )

            processed_pdfs.add(pdf_id)
            continue

        # --------------------------------------------------
        # If current logic decides it does not exist, create vectorstore
        # --------------------------------------------------
        print(f"[DEBUG] CURRENT LOGIC DECISION: Create vectorstore for PDF: {pdf_name}")
        print(f"[DEBUG] Calling create_vectorstore_for_pdf(...) for {pdf_name}")

        try:
            create_vectorstore_for_pdf(pdf)
            print(f"[DEBUG] Vector store created for PDF: {pdf_name}")
            processed_pdfs.add(pdf_id)
        except Exception as e:
            print(f"[ERROR] Failed to create vector store for PDF {pdf_name}: {e}")

    print("\n================ ensure_vectorstores_for_all_pdfs END ================\n")

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
    Determine whether the user wants booklet/PDF links.
    Returns True if:
      1) query clearly looks like booklet/pdf request via regex/keywords, OR
      2) model says YES
    """
    query_lower = query.lower().strip()

    # ---------------------------------------------------
    # 1. Strong keyword-based shortcut
    # ---------------------------------------------------
    pdf_keywords = [
        "pdf",
        "pdfs",
        "booklet",
        "booklets",
        "worksheet",
        "worksheets",
        "book",
        "books",
        "material",
        "materials",
        "notes",
        "document",
        "documents"
    ]

    if any(word in query_lower for word in pdf_keywords):
        # If they are also asking about term/year/week,
        # treat it as a booklet/PDF fetch request.
        if any(x in query_lower for x in ["term", "year", "week"]):
            return True

    # ---------------------------------------------------
    # 2. Pattern-based checks
    # ---------------------------------------------------
    patterns = [
        r"term\s*\d+\s*week\s*\d+",
        r"term\s*\d+",
        r"year\s*\d+"
    ]

    if any(re.search(pattern, query_lower) for pattern in patterns):
        if any(word in query_lower for word in ["booklet", "booklets", "pdf", "pdfs", "worksheet", "worksheets"]):
            return True

    # ---------------------------------------------------
    # 3. OpenAI classifier fallback
    # ---------------------------------------------------
    prompt = (
        "You are a strict intent classifier.\n\n"
        "Decide whether the user is asking to fetch or open booklet/PDF/document links "
        "for study material.\n"
        "Examples that should be YES:\n"
        "- give me year 5 term 1 booklet\n"
        "- fetch term 2 pdfs\n"
        "- show me week 3 worksheet\n"
        "- open the term 1 booklet\n\n"
        "Respond only with YES or NO.\n\n"
        f"Query: {query}"
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

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
        return False





def generate_drive_pdf_url(file_id: str) -> str:
        return f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"

def is_educational_query_openai(query: str, user_id: str, db: Session) -> bool:
    """
    Returns True if the query is educational based on:
    - Local keywords loaded from DB (singleton list)
    - OpenAI classification (with keyword list included in prompt)

    Logs OpenAI usage.
    """

    query_lower = query.lower()

    # -------------------- Load keywords from DB --------------------
    record = db.query(RelevantWords).first()
    educational_keywords = record.singleton.get("educational_keywords", []) if record else []


    # Normalize keywords
    educational_keywords = [w.lower() for w in educational_keywords]

    # -------------------- Quick local checks --------------------
    if any(word in query_lower for word in ["year", "term"]):
        return True

    if any(word in query_lower for word in educational_keywords):
        return True

    # -------------------- Prepare OpenAI prompt --------------------
    keywords_str = ", ".join(educational_keywords)

    prompt = (
        "You are an assistant that classifies queries as educational or not.\n"
        "A query is considered educational if it relates to:\n"
        "- School subjects\n"
        "- Lessons or concepts\n"
        "- Homework or assignments\n"
        "- Quizzes, tests, or academic exercises\n"
        "- General learning or studying topics\n\n"

        f"Use these user-defined educational keywords as guidance:\n"
        f"{keywords_str}\n\n"

        "Respond ONLY with 'Yes' or 'No'.\n\n"
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

        call_cost = calculate_openai_cost(
            model_name="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            multiplier=1.0
        )

        print(f"[INFO] OpenAI API cost for this call: ${call_cost}")

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
    api_key=os.getenv("OPENAI_API_KEY")
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

def normalize_pdf_name(name: str) -> str:
    """
    Normalize PDF names/paths for robust matching:
    - Lowercase
    - Strip extra spaces
    - Replace double extensions or double dots
    - Replace hyphens consistently
    """
    name = name.lower().strip()
    name = name.replace(".pdf.pdf", ".pdf")
    name = name.replace("..", ".")
    name = name.replace("- ", "-")
    name = name.replace("  ", " ")
    return name


def matches_class(name_or_path: str, class_names: list[str]) -> bool:
    """Return True if any class_name is substring of normalized name_or_path (case-insensitive)"""
    norm_str = normalize_pdf_name(name_or_path)
    return any(normalize_pdf_name(cls) in norm_str for cls in class_names)

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

def get_subfolders(parent_id: str):
    try:
        response = drive_service.files().list(
            q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder'",
            spaces="drive",
            fields="files(id, name)"
        ).execute()
        return response.get("files", [])
    except HttpError as e:
        print(f"ERROR: Failed fetching subfolders for parent {parent_id}: {e}")
        return []

#this endpoint removes goole drive access of all students and delete all students in the users table
@app.post("/reset-students")
def reset_students_endpoint(db: Session = Depends(get_db)):
    """
    Reset all students:
      - Remove Drive access from all Year → Term folders
      - Delete students from DB (except Admin)
    Returns a JSON response compatible with front end fetch.
    """
    response = {"status": "success", "message": "", "details": []}

    print("==== Starting reset students process ====")

    # 1. Fetch all users except Admin
    students = db.query(User).filter(User.name != "Admin").all()
    if not students:
        print("No students found to reset.")
        response["status"] = "no_students"
        response["message"] = "No students found to reset."
        return response

    print(f"Found {len(students)} student(s) to reset.")

    # 2. Build Year → Term → folder map using global DEMO_FOLDER_ID
    year_folders = get_subfolders(DEMO_FOLDER_ID)
    folder_map = {}
    for year in year_folders:
        year_name = year["name"].strip().lower()
        term_folders = get_subfolders(year["id"])
        folder_map[year_name] = {term["name"].strip().lower(): term["id"] for term in term_folders}

    # 3. Fetch current term
    current_term_record = db.query(CurrentTerm).first()
    if not current_term_record:
        print("ERROR: No current term found in DB. Exiting.")
        response["status"] = "error"
        response["message"] = "No current term found in DB."
        return response
    current_term = current_term_record.term_name.strip().lower()

    # 4. Remove Drive permissions
    for student in students:
        student_result = {"email": student.email, "removed": [], "skipped": []}

        if not student.class_name:
            print(f"Skipping {student.email}: class_name is empty.")
            student_result["skipped"].append("Empty class_name")
            response["details"].append(student_result)
            continue

        user_years = [c.strip().lower() for c in student.class_name.split(",")]
        print(f"Processing {student.email} for classes {user_years}")

        for year_key in user_years:
            if year_key not in folder_map:
                print(f"WARNING: Year folder '{year_key}' not found in Drive.")
                student_result["skipped"].append(f"Year folder '{year_key}' not found")
                continue
            if current_term not in folder_map[year_key]:
                print(f"WARNING: Term '{current_term}' not found under Year '{year_key}'.")
                student_result["skipped"].append(f"Term '{current_term}' not found")
                continue

            folder_id_to_remove = folder_map[year_key][current_term]

            try:
                permissions = drive_service.permissions().list(
                    fileId=folder_id_to_remove,
                    fields="permissions(id, emailAddress)"
                ).execute().get("permissions", [])

                for perm in permissions:
                    if perm.get("emailAddress") == student.email:
                        drive_service.permissions().delete(
                            fileId=folder_id_to_remove,
                            permissionId=perm["id"]
                        ).execute()
                        print(f"Removed Drive access for {student.email} from folder {folder_id_to_remove}")
                        student_result["removed"].append(folder_id_to_remove)
                        break
                else:
                    student_result["skipped"].append(f"No permission found in folder {folder_id_to_remove}")

            except HttpError as e:
                print(f"Failed to remove Drive access for {student.email} in folder {folder_id_to_remove}: {e}")
                student_result["skipped"].append(f"Failed to remove permission: {e}")

        response["details"].append(student_result)

    # 5. Delete students from DB (except Admin)
    deleted = db.execute(delete(User).where(User.name != "Admin"))
    db.commit()
    print(f"Deleted {deleted.rowcount} student(s) from the database.")
    response["message"] = f"Deleted {deleted.rowcount} student(s) and removed Drive permissions."

    print("==== Reset students process completed successfully ====")
    return response


def get_cached_allowed_pdfs(student_id: str):
    entry = ALLOWED_PDFS_CACHE.get(student_id)

    if not entry:
        return None

    age = time.time() - entry["cached_at"]
    if age > ALLOWED_PDFS_CACHE_TTL_SECONDS:
        ALLOWED_PDFS_CACHE.pop(student_id, None)
        return None

    return entry["data"]


def set_cached_allowed_pdfs(student_id: str, data: dict):
    ALLOWED_PDFS_CACHE[student_id] = {
        "data": data,
        "cached_at": time.time()
    }

def get_allowed_pdfs_for_student(student_id: str, db: Session):
    """
    Returns the list of PDFs a student is allowed to access
    based on:
        Class -> Year -> Current Term

    Returns:
        {
            "student_id": ...,
            "class_name": ...,
            "student_year": ...,
            "current_term": ...,
            "folder_id": ...,
            "files": [
                {"id": "...", "name": "...", "mimeType": "..."},
                ...
            ]
        }
    """

    print("\n========== GET ALLOWED PDFS START ==========")
    print(f"Requested student_id: {student_id}")

    if db is None:
        raise ValueError("A database session must be provided via `db` argument.")

    # ------------------------------------------------
    # 0. Check cache first
    # ------------------------------------------------
    cached_data = get_cached_allowed_pdfs(student_id)
    if cached_data is not None:
        print(f"[ALLOWED PDF CACHE HIT] student_id={student_id}")
        print("========== GET ALLOWED PDFS END ==========\n")
        return cached_data

    print(f"[ALLOWED PDF CACHE MISS] student_id={student_id}")

    # -------------------------
    # 1. Fetch student
    # -------------------------
    student = (
        db.query(Student)
        .filter(Student.student_id == student_id)
        .first()
    )

    if not student:
        print(f"ERROR: No student found for student_id={student_id}")
        print("========== GET ALLOWED PDFS END ==========\n")
        return None

    print("DEBUG: Student found:")
    print(f"    student_id   = {student.student_id}")
    print(f"    name         = {student.name}")
    print(f"    parent_email = {student.parent_email}")
    print(f"    class_name   = {student.class_name}")
    print(f"    student_year = {student.student_year}")

    if not student.class_name:
        print("ERROR: Student class_name is empty.")
        print("========== GET ALLOWED PDFS END ==========\n")
        return None

    if not student.student_year:
        print("ERROR: Student student_year is empty.")
        print("========== GET ALLOWED PDFS END ==========\n")
        return None

    # -------------------------
    # 2. Load current term
    # -------------------------
    current_term_record = db.query(CurrentTerm).first()
    if not current_term_record:
        print("ERROR: No current term found in DB.")
        print("========== GET ALLOWED PDFS END ==========\n")
        return None

    current_term = current_term_record.term_name.strip().lower()
    print(f"DEBUG: Current term from DB = '{current_term}'")

    # -------------------------
    # 3. Normalize student values
    # -------------------------
    student_classes = [
        c.strip().lower()
        for c in student.class_name.split(",")
        if c.strip()
    ]
    student_year = student.student_year.strip().lower()

    print(f"DEBUG: Normalized student classes = {student_classes}")
    print(f"DEBUG: Normalized student year    = {student_year}")

    # -------------------------
    # 4. Helper to fetch subfolders
    # -------------------------
    def get_subfolders(parent_id: str):
        try:
            response = drive_service.files().list(
                q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                spaces="drive",
                fields="files(id, name)"
            ).execute()
            return response.get("files", [])
        except HttpError as e:
            print(f"ERROR: Failed fetching subfolders for parent {parent_id}: {e}")
            return []

    # -------------------------
    # 5. Build Class -> Year -> Term -> folderId map
    # -------------------------
    class_folders = get_subfolders(DEMO_FOLDER_ID)
    print(f"DEBUG: Found Class folders: {[f['name'] for f in class_folders]}")

    folder_map = {}

    for class_folder in class_folders:
        class_name = class_folder["name"].strip().lower()
        class_id = class_folder["id"]

        year_folders = get_subfolders(class_id)
        folder_map[class_name] = {}

        for year_folder in year_folders:
            year_name = year_folder["name"].strip().lower()
            year_id = year_folder["id"]

            term_folders = get_subfolders(year_id)

            folder_map[class_name][year_name] = {
                term["name"].strip().lower(): term["id"]
                for term in term_folders
            }

    print("DEBUG: Folder map:")
    for class_name, years in folder_map.items():
        print(f"  CLASS: {class_name}")
        for year_name, terms in years.items():
            print(f"    YEAR: {year_name}")
            print(f"      TERMS: {list(terms.keys())}")

    # -------------------------
    # 6. Resolve allowed folder
    # -------------------------
    resolved_folder_id = None
    resolved_class = None

    for class_key in student_classes:
        print(f"DEBUG: Checking class '{class_key}' for student...")

        if class_key not in folder_map:
            print(
                f"WARNING: Class folder '{class_key}' not found in Drive. "
                f"Available class folders: {list(folder_map.keys())}"
            )
            continue

        if student_year not in folder_map[class_key]:
            print(
                f"WARNING: Year folder '{student_year}' not found under class '{class_key}'. "
                f"Available years: {list(folder_map[class_key].keys())}"
            )
            continue

        if current_term not in folder_map[class_key][student_year]:
            print(
                f"WARNING: Term '{current_term}' not found under "
                f"class '{class_key}' -> year '{student_year}'."
            )
            continue

        resolved_folder_id = folder_map[class_key][student_year][current_term]
        resolved_class = class_key
        break

    if not resolved_folder_id:
        print("ERROR: Could not resolve a valid folder for this student.")
        print("========== GET ALLOWED PDFS END ==========\n")
        return None

    print("DEBUG: Resolved student folder:")
    print(f"    class     = {resolved_class}")
    print(f"    year      = {student_year}")
    print(f"    term      = {current_term}")
    print(f"    folder_id = {resolved_folder_id}")

    # -------------------------
    # 7. List PDFs inside resolved folder
    # -------------------------
    try:
        response = drive_service.files().list(
            q=(
                f"'{resolved_folder_id}' in parents "
                f"and mimeType='application/pdf' "
                f"and trashed=false"
            ),
            spaces="drive",
            fields="files(id, name, mimeType)"
        ).execute()

        pdf_files = response.get("files", [])

    except HttpError as e:
        print("ERROR: Failed to list PDFs inside resolved folder.")
        print(str(e))
        print("========== GET ALLOWED PDFS END ==========\n")
        return None

    print("DEBUG: PDFs found in allowed folder:")
    if not pdf_files:
        print("    No PDFs found.")
    else:
        for pdf in pdf_files:
            print(
                f"    file_name={pdf.get('name')}, "
                f"file_id={pdf.get('id')}, "
                f"mimeType={pdf.get('mimeType')}"
            )

    result = {
        "student_id": student.student_id,
        "class_name": student.class_name,
        "student_year": student.student_year,
        "current_term": current_term_record.term_name,
        "folder_id": resolved_folder_id,
        "files": pdf_files
    }

    # ------------------------------------------------
    # 8. Save in cache before returning
    # ------------------------------------------------
    set_cached_allowed_pdfs(student_id, result)
    print(f"[ALLOWED PDF CACHE STORE] student_id={student_id}")

    print("========== GET ALLOWED PDFS END ==========\n")
    return result

@app.get("/debug/student-allowed-pdfs")
def debug_student_allowed_pdfs(
    student_id: str,
    db: Session = Depends(get_db)
):
    result = get_allowed_pdfs_for_student(student_id=student_id, db=db)

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Could not resolve allowed PDFs for this student."
        )

    return result
def generate_backend_pdf_url(student_id: str, file_id: str) -> str:
    query = urlencode({
        "student_id": student_id,
        "file_id": file_id
    })
    return f"{BACKEND_PUBLIC_BASE_URL}/student-pdf?{query}"

def extract_drive_file_id(url: str) -> str | None:
    if not url:
        return None

    patterns = [
        r"/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"/folders/([a-zA-Z0-9_-]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None
import io
import fitz
from fastapi.responses import StreamingResponse
PAGE_IMAGE_CACHE = {}
PAGE_IMAGE_CACHE_TTL_SECONDS = 60 * 10   # 10 minutes


def get_cached_page_image(file_id: str, page: int):
    cache_key = f"{file_id}:{page}"
    entry = PAGE_IMAGE_CACHE.get(cache_key)

    if not entry:
        return None

    age = time.time() - entry["cached_at"]
    if age > PAGE_IMAGE_CACHE_TTL_SECONDS:
        PAGE_IMAGE_CACHE.pop(cache_key, None)
        return None

    return entry["image_bytes"]


def set_cached_page_image(file_id: str, page: int, image_bytes: bytes):
    cache_key = f"{file_id}:{page}"
    PAGE_IMAGE_CACHE[cache_key] = {
        "image_bytes": image_bytes,
        "cached_at": time.time()
    }
def get_cached_pdf_bytes(file_id: str):
    entry = PDF_CACHE.get(file_id)

    if not entry:
        return None

    age = time.time() - entry["cached_at"]
    if age > PDF_CACHE_TTL_SECONDS:
        PDF_CACHE.pop(file_id, None)
        return None

    return entry["pdf_bytes"]


def set_cached_pdf_bytes(file_id: str, pdf_bytes: bytes):
    PDF_CACHE[file_id] = {
        "pdf_bytes": pdf_bytes,
        "cached_at": time.time()
    }

@app.get("/student-pdf-page")
async def student_pdf_page(
    student_id: str,
    file_id: str,
    page: int = 1,
    db: Session = Depends(get_db)
):
    print("\n========== STUDENT PDF PAGE START ==========")
    print(f"student_id = {student_id}")
    print(f"file_id    = {file_id}")
    print(f"page       = {page}")

    allowed_data = get_allowed_pdfs_for_student(student_id=student_id, db=db)

    if not allowed_data:
        raise HTTPException(
            status_code=404,
            detail="No allowed PDFs found for this student."
        )

    pdf_list = allowed_data.get("files", [])
    allowed_file_ids = {pdf["id"] for pdf in pdf_list}
    print("[DEBUG PAGE] allowed_file_ids =", allowed_file_ids)
    print("[DEBUG PAGE] requested file_id =", file_id)

    if file_id not in allowed_file_ids:
        print("ACCESS DENIED: file not allowed for this student")
        raise HTTPException(
            status_code=403,
            detail="You are not allowed to access this PDF."
        )

    try:
        # -----------------------------
        # STEP A: check rendered page cache first
        # -----------------------------
        cached_image = get_cached_page_image(file_id, page)
        if cached_image is not None:
            print(f"[PAGE CACHE HIT] Returning cached page image for file={file_id}, page={page}")
            print("========== STUDENT PDF PAGE END ==========\n")
            return StreamingResponse(
                io.BytesIO(cached_image),
                media_type="image/png"
            )

        print(f"[PAGE CACHE MISS] Rendering page {page} for file={file_id}")

        # -----------------------------
        # STEP B: get cached PDF bytes
        # -----------------------------
        pdf_bytes = get_cached_pdf_bytes(file_id)

        if pdf_bytes is None:
            print(f"[PDF CACHE MISS] Downloading file {file_id} from Drive for page render")
            request = drive_service.files().get_media(fileId=file_id)
            pdf_bytes = request.execute()
            set_cached_pdf_bytes(file_id, pdf_bytes)
        else:
            print(f"[PDF CACHE HIT] Using cached PDF for page render: {file_id}")

        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if page < 1 or page > len(pdf_doc):
            raise HTTPException(
                status_code=404,
                detail="Requested page does not exist."
            )

        # -----------------------------
        # STEP C: render page once
        # -----------------------------
        pdf_page = pdf_doc.load_page(page - 1)
        pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image_bytes = pix.tobytes("png")

        # -----------------------------
        # STEP D: cache rendered page image
        # -----------------------------
        set_cached_page_image(file_id, page, image_bytes)

        print("ACCESS GRANTED: returning newly rendered page image")
        print("========== STUDENT PDF PAGE END ==========\n")

        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR while rendering PDF page: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to render PDF page."
        )

@app.get("/student-pdf-search")
async def student_pdf_search(
    student_id: str,
    file_id: str,
    query: str,
    db: Session = Depends(get_db)
):
    """
    Search inside a protected PDF that the student is allowed to access.
    Returns matching page numbers with small text snippets.
    """
    print("\n========== STUDENT PDF SEARCH START ==========")
    print(f"student_id = {student_id}")
    print(f"file_id    = {file_id}")
    print(f"query      = {query}")

    try:
        # --------------------------------------------------
        # Step 1: Validate query
        # --------------------------------------------------
        query = (query or "").strip()
        if not query:
            print("[WARNING] Empty search query received.")
            return JSONResponse({
                "query": query,
                "matches": []
            })

        # --------------------------------------------------
        # Step 2: Check student access to this PDF
        # --------------------------------------------------
        allowed_data = get_allowed_pdfs_for_student(student_id=student_id, db=db)

        if not allowed_data:
            print("[ERROR] No allowed PDFs found for this student.")
            raise HTTPException(
                status_code=404,
                detail="No allowed PDFs found for this student."
            )

        pdf_list = allowed_data.get("files", [])
        allowed_file_ids = {pdf["id"] for pdf in pdf_list}

        if file_id not in allowed_file_ids:
            print("[ERROR] Access denied: file not allowed for this student.")
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to search this booklet."
            )

        print("[DEBUG] Access granted for PDF search.")

        # --------------------------------------------------
        # Step 3: Load PDF bytes (cache first, then Drive)
        # --------------------------------------------------
        pdf_bytes = get_cached_pdf_bytes(file_id)

        if pdf_bytes is None:
            print(f"[PDF CACHE MISS] Downloading file {file_id} from Drive for PDF search")
            request = drive_service.files().get_media(fileId=file_id)
            pdf_bytes = request.execute()
            set_cached_pdf_bytes(file_id, pdf_bytes)
        else:
            print(f"[PDF CACHE HIT] Using cached PDF for PDF search: {file_id}")

        # --------------------------------------------------
        # Step 4: Open PDF and search page by page
        # --------------------------------------------------
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(pdf_doc)

        query_lower = query.lower()
        matches = []

        print(f"[DEBUG] Searching across {total_pages} pages...")

        for page_idx in range(total_pages):
            page_number = page_idx + 1

            try:
                page = pdf_doc[page_idx]
                page_text = page.get_text("text") or ""
                page_text_clean = page_text.replace("\x00", " ").strip()

                if not page_text_clean:
                    continue

                page_text_lower = page_text_clean.lower()

                if query_lower in page_text_lower:
                    # Build a small snippet around the first match
                    match_pos = page_text_lower.find(query_lower)

                    snippet_start = max(0, match_pos - 120)
                    snippet_end = min(len(page_text_clean), match_pos + len(query) + 120)

                    snippet = page_text_clean[snippet_start:snippet_end]
                    snippet = snippet.replace("\n", " ").replace("  ", " ").strip()

                    matches.append({
                        "page": page_number,
                        "snippet": snippet
                    })

                    print(f"[MATCH] Found match on page {page_number}")
                    print(f"        snippet = {snippet[:300]}")

            except Exception as page_err:
                print(f"[WARNING] Failed searching page {page_number}: {page_err}")

        print(f"[DEBUG] Search complete. Total matches = {len(matches)}")
        print("========== STUDENT PDF SEARCH END ==========\n")

        return JSONResponse({
            "query": query,
            "matches": matches
        })

    except HTTPException as http_err:
        print(f"[HTTP ERROR] {http_err.detail}")
        print("========== STUDENT PDF SEARCH FAILED ==========\n")
        raise http_err

    except Exception as e:
        print(f"[ERROR] Failed PDF search: {e}")
        import traceback
        traceback.print_exc()
        print("========== STUDENT PDF SEARCH FAILED ==========\n")
        raise HTTPException(
            status_code=500,
            detail="Failed to search inside the booklet."
        )
@app.get("/student-pdf-meta")
async def student_pdf_meta(
    student_id: str,
    file_id: str,
    db: Session = Depends(get_db)
):
    print("\n========== STUDENT PDF META ==========")
    print(f"student_id = {student_id}")
    print(f"file_id    = {file_id}")

    # ---------------- Permission check ----------------
    allowed_data = get_allowed_pdfs_for_student(
        student_id=student_id,
        db=db
    )

    if not allowed_data:
        raise HTTPException(
            status_code=404,
            detail="No allowed PDFs found for this student."
        )

    pdf_list = allowed_data.get("files", [])
    allowed_file_ids = {pdf["id"] for pdf in pdf_list}
    print("[DEBUG META] allowed_file_ids =", allowed_file_ids)
    print("[DEBUG META] requested file_id =", file_id)

    if file_id not in allowed_file_ids:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to access this booklet."
        )

    # ---------------- Load PDF ----------------
    try:
        pdf_bytes = get_cached_pdf_bytes(file_id)

        if pdf_bytes is None:
            print(f"[PDF CACHE MISS] Downloading file {file_id}")
            request = drive_service.files().get_media(fileId=file_id)
            pdf_bytes = request.execute()
            set_cached_pdf_bytes(file_id, pdf_bytes)
        else:
            print(f"[PDF CACHE HIT] Using cached PDF {file_id}")

        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        return {
            "file_id": file_id,
            "total_pages": len(pdf_doc)
        }

    except Exception as e:
        print(f"[ERROR] Failed loading PDF metadata: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load PDF metadata."
        )

@app.get("/student-pdf")
async def student_pdf(
    student_id: str,
    file_id: str,
    page: int = 1,
    db: Session = Depends(get_db)
):
    print("\n========== STUDENT PDF ACCESS START ==========")
    print(f"student_id = {student_id}")
    print(f"file_id    = {file_id}")

    allowed_data = get_allowed_pdfs_for_student(student_id=student_id, db=db)

    if not allowed_data:
        raise HTTPException(
            status_code=404,
            detail="No allowed PDFs found for this student."
        )

    pdf_list = allowed_data.get("files", [])
    allowed_file_ids = {pdf["id"] for pdf in pdf_list}

    if file_id not in allowed_file_ids:
        print("ACCESS DENIED: file not allowed for this student")

        return HTMLResponse(
            content=f"""
            <html>
            <head>
                <title>Access Denied</title>
                <style>
                    body {{
                        margin: 0;
                        padding: 0;
                        font-family: Arial, sans-serif;
                        background: #f7f7f7;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        min-height: 100vh;
                    }}

                    .access-card {{
                        max-width: 700px;
                        width: 90%;
                        background: white;
                        border-radius: 16px;
                        padding: 32px;
                        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
                        text-align: center;
                    }}

                    .icon {{
                        font-size: 64px;
                        margin-bottom: 16px;
                    }}

                    .title {{
                        font-size: 32px;
                        font-weight: 700;
                        color: #dc2626;
                        margin-bottom: 12px;
                    }}

                    .message {{
                        font-size: 18px;
                        color: #333;
                        line-height: 1.6;
                        margin-bottom: 24px;
                    }}

                    .subtext {{
                        font-size: 15px;
                        color: #666;
                        line-height: 1.5;
                        margin-bottom: 28px;
                    }}

                    .button {{
                        display: inline-block;
                        padding: 12px 22px;
                        border-radius: 10px;
                        background: #2563eb;
                        color: white;
                        text-decoration: none;
                        font-weight: 600;
                        font-size: 15px;
                    }}

                    .button:hover {{
                        background: #1d4ed8;
                    }}
                    .pdf-search-bar {
                        display: flex;
                        gap: 10px;
                        margin: 16px auto;
                        width: min(900px, 92%);
                        align-items: center;
                    }

                    .pdf-search-bar input {
                        flex: 1;
                        padding: 10px 12px;
                        font-size: 15px;
                        border: 1px solid #ccc;
                        border-radius: 8px;
                        outline: none;
                    }

                    .pdf-search-bar button {
                        padding: 10px 16px;
                        font-size: 15px;
                        border: none;
                        border-radius: 8px;
                        cursor: pointer;
                        background: #f97316;
                        color: white;
                        font-weight: 600;
                    }

                    .pdf-search-results {
                        width: min(900px, 92%);
                        margin: 10px auto 18px auto;
                        display: flex;
                        flex-direction: column;
                        gap: 10px;
                    }

                    .pdf-search-result-item {
                        border: 1px solid #e5e7eb;
                        border-radius: 10px;
                        padding: 12px;
                        background: #fff;
                        cursor: pointer;
                        transition: background 0.2s ease;
                    }

                    .pdf-search-result-item:hover {
                        background: #f9fafb;
                    }

                    .pdf-search-result-page {
                        font-weight: 700;
                        margin-bottom: 6px;
                        color: #111827;
                    }

                    .pdf-search-result-snippet {
                        color: #374151;
                        line-height: 1.5;
                        font-size: 14px;
                    }

                    .pdf-search-empty {
                        color: #6b7280;
                        padding: 8px 2px;
                        font-size: 14px;
                    }
                </style>
            </head>
            <body>
                <div class="access-card">
                    <div class="icon">🔒</div>
                    <div class="title">Access Denied</div>

                    <div class="message">
                        You do not have permission to view this booklet.
                    </div>

                    <div class="subtext">
                        This booklet does not belong to your currently allowed class, year, or term access.
                        If you believe this is a mistake, please contact your teacher or support team.
                    </div>

                    
                </div>
            </body>
            </html>
            """,
            status_code=403
        )

    print("ACCESS GRANTED: opening protected viewer shell")

    try:
        pdf_bytes = get_cached_pdf_bytes(file_id)

        if pdf_bytes is None:
            print(f"[PDF CACHE MISS] Downloading file {file_id} from Drive for viewer shell")
            request = drive_service.files().get_media(fileId=file_id)
            pdf_bytes = request.execute()
            set_cached_pdf_bytes(file_id, pdf_bytes)
        else:
            print(f"[PDF CACHE HIT] Using cached PDF for viewer shell: {file_id}")

        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(pdf_doc)
        requested_page = page if page else 1

        if requested_page < 1:
            requested_page = 1
        if requested_page > total_pages:
            requested_page = total_pages

        print(f"[DEBUG] requested_page = {page}")
        print(f"[DEBUG] clamped_page   = {requested_page}")

        print("========== STUDENT PDF ACCESS END ==========\n")

        return HTMLResponse(f"""
        <html>
        <head>
            <title>Protected Booklet Viewer</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: #f7f7f7;
                    margin: 0;
                    padding: 0;
                }}

                .viewer-shell {{
                    max-width: 1000px;
                    margin: 30px auto;
                    background: white;
                    padding: 24px;
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                }}

                .viewer-title {{
                    font-size: 28px;
                    font-weight: 700;
                    margin-bottom: 10px;
                }}

                .viewer-meta {{
                    color: #444;
                    margin-bottom: 20px;
                    line-height: 1.6;
                }}

                .pdf-search-bar {{
                    display: flex;
                    gap: 10px;
                    margin: 18px 0 12px 0;
                    align-items: center;
                    flex-wrap: wrap;
                }}

                .pdf-search-bar input {{
                    flex: 1;
                    min-width: 220px;
                    padding: 10px 12px;
                    font-size: 15px;
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    outline: none;
                }}

                .pdf-search-bar button {{
                    padding: 10px 16px;
                    border: none;
                    border-radius: 8px;
                    background: #f97316;
                    color: white;
                    font-size: 14px;
                    font-weight: 600;
                    cursor: pointer;
                }}

                .pdf-search-results {{
                    margin: 10px 0 20px 0;
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }}

                .pdf-search-result-item {{
                    border: 1px solid #e5e7eb;
                    border-radius: 10px;
                    padding: 12px;
                    background: #fff;
                    cursor: pointer;
                    transition: background 0.2s ease;
                }}

                .pdf-search-result-item:hover {{
                    background: #f9fafb;
                }}

                .pdf-search-result-page {{
                    font-weight: 700;
                    margin-bottom: 6px;
                    color: #111827;
                }}

                .pdf-search-result-snippet {{
                    color: #374151;
                    line-height: 1.5;
                    font-size: 14px;
                }}

                .pdf-search-empty {{
                    color: #6b7280;
                    padding: 8px 2px;
                    font-size: 14px;
                }}

                .controls {{
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    margin-top: 16px;
                    margin-bottom: 12px;
                    flex-wrap: wrap;
                }}

                .controls button {{
                    padding: 10px 18px;
                    border: none;
                    border-radius: 8px;
                    background: #2563eb;
                    color: white;
                    font-size: 14px;
                    cursor: pointer;
                }}

                .controls button:disabled {{
                    background: #9ca3af;
                    cursor: not-allowed;
                }}

                .page-status {{
                    font-weight: 600;
                    color: #222;
                }}

                .jump-box {{
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin-left: auto;
                    flex-wrap: wrap;
                }}

                .jump-box input {{
                    width: 90px;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    font-size: 14px;
                }}

                .jump-box button {{
                    padding: 10px 16px;
                    border: none;
                    border-radius: 8px;
                    background: #16a34a;
                    color: white;
                    font-size: 14px;
                    cursor: pointer;
                }}

                .page-image {{
                    width: 100%;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    margin-top: 16px;
                }}
            </style>
        </head>
        <body>
            <div class="viewer-shell">
                <div class="viewer-title">Protected Booklet Viewer</div>

                <div class="viewer-meta">
                    Student: <b>{student_id}</b><br/>
                    File ID: <b>{file_id}</b><br/>
                    Total pages: <b>{total_pages}</b>
                </div>

                <!-- Search Bar -->
                <div class="pdf-search-bar">
                    <input
                        type="text"
                        id="pdfSearchInput"
                        placeholder="Search inside this booklet..."
                    />
                    <button id="pdfSearchBtn">Search</button>
                </div>

                <!-- Search Results -->
                <div id="pdfSearchResults" class="pdf-search-results"></div>

                <!-- Page Controls -->
                <div class="controls">
                    <button id="prevBtn">Previous</button>
                    <span class="page-status" id="pageStatus">Page {requested_page} of {total_pages}</span>
                    <button id="nextBtn">Next</button>

                    <div class="jump-box">
                        <label for="pageInput"><b>Go to page:</b></label>
                        <input
                            id="pageInput"
                            type="number"
                            min="1"
                            max="{total_pages}"
                            placeholder="Page #"
                        />
                        <button id="goBtn">Go</button>
                    </div>
                </div>

                <!-- Page Image -->
                <img
                    id="pdfPageImage"
                    class="page-image"
                    src="/student-pdf-page?student_id={student_id}&file_id={file_id}&page={requested_page}"
                    alt="Booklet page"
                />
            </div>

            <script>
                const studentId = "{student_id}";
                const fileId = "{file_id}";
                const totalPages = {total_pages};

                let currentPage = {requested_page};

                const img = document.getElementById("pdfPageImage");
                const prevBtn = document.getElementById("prevBtn");
                const nextBtn = document.getElementById("nextBtn");
                const pageStatus = document.getElementById("pageStatus");
                const pageInput = document.getElementById("pageInput");
                const goBtn = document.getElementById("goBtn");

                const pdfSearchInput = document.getElementById("pdfSearchInput");
                const pdfSearchBtn = document.getElementById("pdfSearchBtn");
                const pdfSearchResults = document.getElementById("pdfSearchResults");

                function updateViewer() {{
                    img.src = `/student-pdf-page?student_id=${{studentId}}&file_id=${{fileId}}&page=${{currentPage}}`;
                    pageStatus.textContent = `Page ${{currentPage}} of ${{totalPages}}`;

                    prevBtn.disabled = currentPage === 1;
                    nextBtn.disabled = currentPage === totalPages;
                    pageInput.value = currentPage;
                }}

                function jumpToPage(pageNumber) {{
                    let targetPage = parseInt(pageNumber, 10);

                    if (isNaN(targetPage)) {{
                        return;
                    }}

                    if (targetPage < 1) {{
                        targetPage = 1;
                    }}

                    if (targetPage > totalPages) {{
                        targetPage = totalPages;
                    }}

                    currentPage = targetPage;
                    updateViewer();
                }}

                function escapeHtml(text) {{
                    return text
                        .replace(/&/g, "&amp;")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;")
                        .replace(/"/g, "&quot;")
                        .replace(/'/g, "&#039;");
                }}

                async function searchInsidePdf() {{
                    const query = pdfSearchInput.value.trim();

                    if (!query) {{
                        pdfSearchResults.innerHTML = '<div class="pdf-search-empty">Please enter something to search.</div>';
                        return;
                    }}

                    pdfSearchResults.innerHTML = '<div class="pdf-search-empty">Searching...</div>';

                    try {{
                        const res = await fetch(
                            `/student-pdf-search?student_id=${{encodeURIComponent(studentId)}}&file_id=${{encodeURIComponent(fileId)}}&query=${{encodeURIComponent(query)}}`
                        );

                        if (!res.ok) {{
                            throw new Error(`Search request failed with status ${{res.status}}`);
                        }}

                        const data = await res.json();
                        const matches = data.matches || [];

                        if (matches.length === 0) {{
                            pdfSearchResults.innerHTML = '<div class="pdf-search-empty">No matches found.</div>';
                            return;
                        }}

                        pdfSearchResults.innerHTML = matches.map(match => `
                            <div class="pdf-search-result-item" data-page="${{match.page}}">
                                <div class="pdf-search-result-page">Page ${{match.page}}</div>
                                <div class="pdf-search-result-snippet">${{escapeHtml(match.snippet || "")}}</div>
                            </div>
                        `).join("");

                        document.querySelectorAll(".pdf-search-result-item").forEach(item => {{
                            item.addEventListener("click", () => {{
                                const page = parseInt(item.dataset.page, 10);
                                if (!Number.isNaN(page)) {{
                                    jumpToPage(page);
                                }}
                            }});
                        }});

                    }} catch (err) {{
                        console.error("PDF search failed:", err);
                        pdfSearchResults.innerHTML = '<div class="pdf-search-empty">Search failed. Please try again.</div>';
                    }}
                }}

                prevBtn.addEventListener("click", () => {{
                    if (currentPage > 1) {{
                        currentPage -= 1;
                        updateViewer();
                    }}
                }});

                nextBtn.addEventListener("click", () => {{
                    if (currentPage < totalPages) {{
                        currentPage += 1;
                        updateViewer();
                    }}
                }});

                goBtn.addEventListener("click", () => {{
                    jumpToPage(pageInput.value);
                }});

                pageInput.addEventListener("keydown", (event) => {{
                    if (event.key === "Enter") {{
                        jumpToPage(pageInput.value);
                    }}
                }});

                pdfSearchBtn.addEventListener("click", searchInsidePdf);

                pdfSearchInput.addEventListener("keydown", (event) => {{
                    if (event.key === "Enter") {{
                        searchInsidePdf();
                    }}
                }});

                updateViewer();
            </script>
        </body>
        </html>
        """)

    except Exception as e:
        print(f"ERROR while preparing viewer shell: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load booklet viewer."
        )
def get_or_create_chatbot_conversation(db, conversation_uuid: str, student: Student):
    conversation = (
        db.query(ChatbotConversation)
        .filter(ChatbotConversation.conversation_uuid == conversation_uuid)
        .first()
    )

    if conversation:
        return conversation

    now_au = australia_now()

    conversation = ChatbotConversation(
        conversation_uuid=conversation_uuid,
        student_id=student.student_id,
        student_name=student.name,
        parent_email=student.parent_email,
        class_name=student.class_name,
        student_year=student.student_year,
        center_code=student.center_code,
        center_name=student.center_name,
        started_at=now_au,
        last_message_at=now_au,
        created_at=now_au,
        updated_at=now_au,
        message_count=0,
        status="active"
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation

def save_chatbot_message(
    db,
    conversation_id: int,
    role: str,
    message_text: str,
    source_name: str = None,
    reasoning_level: str = None,
    class_name: str = None,
    pdf_name: str = None,
    pdf_page: int = None,
    pdf_file_id: str = None,
    response_links=None
):
    msg = ChatbotMessage(
        conversation_id=conversation_id,
        role=role,
        message_text=message_text,
        source_name=source_name,
        reasoning_level=reasoning_level,
        class_name=class_name,
        pdf_name=pdf_name,
        pdf_page=pdf_page,
        pdf_file_id=pdf_file_id,
        response_links=response_links,
        created_at=australia_now()
    )
    db.add(msg)
    return msg


@app.get("/search")
async def search_pdfs(
    query: str,
    reasoning: str,
    user_id: str,
    conversation_uuid: str,
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
        return JSONResponse({
            "source_name": "Academy Answer",
            "answer_markdown": f"You have reached the maximum of {MAX_INTERACTIONS} interactions. Please contact support for further queries.",
            "links": []
        })

    #------------------ Step 0b: Check educational query ------------------
    #if not is_educational_query_openai(query, user_id=user_id, db=db):
     #   return JSONResponse([{
      #      "name": "**Academy Answer**",
       #     "snippet": "Your query does not seem to be educational or relevant.",
        #    "links": []
       # }])
    student = (
        db.query(Student)
        .filter(Student.name == user_id)
        .first()
    )

    if not student:
        print(f"[ERROR] No student found for chatbot user_id={user_id}")
        return JSONResponse({
            "source_name": "Academy Answer",
            "answer_markdown": "Unable to identify the student account for this request.",
            "links": []
        })
    student_year = student.student_year
    
    conversation = get_or_create_chatbot_conversation(
        db=db,
        conversation_uuid=conversation_uuid,
        student=student
    )
    save_chatbot_message(
        db=db,
        conversation_id=conversation.id,
        role="user",
        message_text=query,
        reasoning_level=reasoning,
        class_name=class_name or student.class_name
    )
    real_student_id = student.student_id

    print("[DEBUG] Student resolved for chatbot request:")
    print(f"        chatbot user_id = {user_id}")
    print(f"        student.name    = {student.name}")
    print(f"        student_id      = {student.student_id}")
    print(f"        class_name DB   = {student.class_name}")
    print(f"        student_year DB = {student.student_year}")
    allowed_pdf_data = get_allowed_pdfs_for_student(real_student_id, db)
    allowed_pdf_files = allowed_pdf_data.get("files", []) if allowed_pdf_data else []

    print("[DEBUG] Allowed PDFs for student:")
    for apdf in allowed_pdf_files:
        print(f"        allowed_pdf name = {apdf.get('name')}, id = {apdf.get('id')}")
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
    #here
    pdf_files = []
    for pdf in all_pdfs:
        pdf_path = pdf.get("path", "")
        
        # If no class filter → accept ALL PDFs
        if not class_names_list:
            pdf_files.append(pdf)
            continue
    
        # Use robust class matching
        if matches_class(pdf_path, class_names_list):
            pdf_files.append(pdf)
        #here
    #-------- commenting out code for conversation context-------
    # context_gist = get_context_gist(user_id)
    # is_first_query = len(user_contexts[user_id]) == 0
    # ----------------------------
    # CONTEXT MODE DISABLED
    # ----------------------------
    #context_gist = ""
    # -------------------------------------------------
    # CONTEXT MODE DISABLED
    # -------------------------------------------------
    use_gpt_only = False
    is_first_query = False

      # -------------------------------------------------
    # CONTEXT MODE DISABLED
    # Query classification temporarily bypassed
    # -------------------------------------------------
    # query_type = classify_query_type(query, context_gist, user_id, pdf_files, db=db)
    # if is_first_query:
    #     query_type = "pdf_only"

    # -------------------------------------------------
    # TEMPORARILY DISABLE CONTEXT-ONLY MODE
    # use_context_only = query_type == "context_only"
    use_context_only = False
    # -------------------------------------------------

    # ------------------ Step 3: Handle PDF link requests ------------------
    print("\n================ PDF REQUEST DEBUG START ================")
    print(f"[DEBUG] Query received for PDF detection: {query}")

    pdf_request_detected = is_pdf_request(query, user_id=user_id, db=db)
    print(f"[DEBUG] is_pdf_request returned: {pdf_request_detected}")
    print(f"[DEBUG] pdf_files count before PDF branch: {len(pdf_files)}")

    pdf_urls_to_send = []
    if pdf_files and pdf_request_detected:
        print("[DEBUG] Entered PDF request branch")

        query_lower = query.lower()
        year_match = re.search(r"year\s*(\d+)", query_lower)
        query_year = year_match.group(1) if year_match else None

        term_match = re.search(r"term\s*(\d+)", query_lower)
        query_term = term_match.group(1) if term_match else None

        week_match = re.search(r"week\s*(\d+)", query_lower)
        query_week = week_match.group(1) if week_match else None

        print(f"[DEBUG] Extracted query_year = {query_year}")
        print(f"[DEBUG] Extracted query_term = {query_term}")
        print(f"[DEBUG] Extracted query_week = {query_week}")

        filtered_pdfs = []
        for pdf in allowed_pdf_files:
            name_lower = pdf.get("name", "").lower()

            if query_year and not (
                f"year {query_year}" in name_lower or
                f"year_{query_year}" in name_lower or
                f"y{query_year}" in name_lower
            ):
                
                print("[DEBUG] Skipped بسبب year mismatch")
                continue

            term_matches = re.findall(r"(?:term|t)\s*[_-]*\s*(\d+)", name_lower)
            week_matches = re.findall(r"(?:week|w)\s*[_-]*\s*(\d+)", name_lower)

            print(f"[DEBUG] term_matches = {term_matches}")
            print(f"[DEBUG] week_matches = {week_matches}")

            if (query_term is None or query_term in term_matches) and (query_week is None or query_week in week_matches):
                print("[DEBUG] PDF matched filters")
                filtered_pdfs.append(pdf)
            else:
                print("[DEBUG] PDF did NOT match term/week filters")

        print(f"[DEBUG] filtered_pdfs count = {len(filtered_pdfs)}")

        if filtered_pdfs:
            pdf_urls_to_send = []   # Keep this for backward compatibility
            pdfs_to_send = []

            for pdf in filtered_pdfs:

                frontend_pdf_url = (
                    f"{FRONTEND_PUBLIC_BASE_URL}/pdf-viewer"
                    f"?student_id={real_student_id}&file_id={pdf['id']}"
                )

                pdf_urls_to_send.append(frontend_pdf_url)

                pdfs_to_send.append({
                    "name": pdf.get("name", "booklet.pdf"),
                    "url": frontend_pdf_url
                })

            answer_text = "Here are the PDFs you requested:"
        else:
            answer_text = "No PDFs found."
            print("[DEBUG] No filtered PDFs found, returning 'No PDFs found.'")

        source_name = "Academy Answer"
        

        append_to_user_context(user_id, "user", query)
        append_to_user_context(user_id, "assistant", answer_text)

        save_chatbot_message(
            db=db,
            conversation_id=conversation.id,
            role="assistant",
            message_text=answer_text,
            source_name="Academy Answer",
            reasoning_level=reasoning,
            class_name=class_name or student.class_name,
            response_links=pdf_urls_to_send
        )

        now_au = australia_now()

        conversation.last_message_at = now_au
        conversation.message_count = (conversation.message_count or 0) + 2
        conversation.updated_at = now_au

        db.commit()

        print("================ PDF REQUEST DEBUG END ================\n")
        return JSONResponse({
            "source_name": "Academy Answer",
            "answer_markdown": answer_text,
            "links": pdf_urls_to_send,   # Existing frontend compatibility
            "pdfs": pdfs_to_send         # New structured PDF data
        })

    print("[DEBUG] PDF branch NOT entered")
    print("================ PDF REQUEST DEBUG END ================\n")
    # ------------------ Step 4: Retrieve top PDF chunks ------------------
    top_chunks = []
    raw_top_chunks = []
    fallback_reason = None

    print("\n================ ACADEMY RETRIEVAL DEBUG START ================")
    print(f"[DEBUG] Query = {query}")
    print(f"[DEBUG] class_name received = {class_name}")
    print(f"[DEBUG] pdf_files count = {len(pdf_files)}")
    print(f"[DEBUG] use_context_only = {use_context_only}")
    print(f"[DEBUG] FAISS_INDEX is None? {FAISS_INDEX is None}")
    print(f"[DEBUG] FAISS_METADATA is None? {FAISS_METADATA is None}")

    faiss_available = FAISS_INDEX is not None and FAISS_METADATA is not None
    print(f"[DEBUG] faiss_available = {faiss_available}")

    if pdf_files and not use_context_only:
        class_list = [cn.strip() for cn in class_name.split(",")] if class_name else []
        print(f"[DEBUG] class_list = {class_list}")

        if faiss_available:
            if FAISS_INDEX is not None:
                print(f"[DEBUG] FAISS_INDEX.ntotal = {FAISS_INDEX.ntotal}")
            if FAISS_METADATA is not None:
                print(f"[DEBUG] len(FAISS_METADATA) = {len(FAISS_METADATA)}")

            # --------------------------------------------------
            # Step 4.1: Convert query to embedding
            # --------------------------------------------------
            try:
                print("[DEBUG] Embedding query for FAISS retrieval...")
                query_embedding = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    openai_api_key=os.environ.get("OPENAI_API_KEY")
                ).embed_query(query)

                query_vector = np.array([query_embedding], dtype="float32")
                faiss.normalize_L2(query_vector)
                print("[DEBUG] Query embedding complete.")
            except Exception as e:
                print(f"[ERROR] Failed to embed query: {e}")
                fallback_reason = "query_embedding_failed"

            # --------------------------------------------------
            # Step 4.2: Search FAISS
            # --------------------------------------------------
            if fallback_reason is None:
                try:
                    k = TOP_K
                    print(f"[DEBUG] Running FAISS search with TOP_K={k} ...")
                    D, I = FAISS_INDEX.search(query_vector, k)
                    print("[DEBUG] FAISS search complete.")
                    print(f"[DEBUG] Raw FAISS distances = {D}")
                    print(f"[DEBUG] Raw FAISS indices   = {I}")
                except Exception as e:
                    print(f"[ERROR] FAISS search failed: {e}")
                    fallback_reason = "faiss_search_failed"

            # --------------------------------------------------
            # Step 4.3: Build raw_top_chunks BEFORE class filter
            # --------------------------------------------------
            if fallback_reason is None:
                raw_top_chunks = []
                for idx, score in zip(I[0], D[0]):
                    if idx < 0 or idx >= len(FAISS_METADATA):
                        continue

                    chunk_meta = FAISS_METADATA[idx].copy()
                    chunk_meta["score"] = float(score)
                    raw_top_chunks.append(chunk_meta)

                print(f"[DEBUG] raw_top_chunks BEFORE class filter: {len(raw_top_chunks)}")

                print("\n================ RAW TOP CHUNKS DEBUG ================")
                print(f"[DEBUG] Query: {query}")
                for i, c in enumerate(raw_top_chunks, start=1):
                    print(f"\n--- RAW TOP CHUNK #{i} ---")
                    print(f"pdf_name    = {c.get('pdf_name')}")
                    print(f"page_number = {c.get('page_number')}")
                    print(f"chunk_index = {c.get('chunk_index')}")
                    print(f"pdf_link    = {c.get('pdf_link')}")
                    print(f"score       = {c.get('score')}")
                    chunk_text = c.get("chunk_text", "")
                    print(f"chunk_text  = {chunk_text[:1500]}")
                print("======================================================\n")

                # --------------------------------------------------
                # TEMP DEBUG: Disable class filtering completely
                # --------------------------------------------------
                top_chunks = raw_top_chunks.copy()
                print("[DEBUG] TEMP: Skipping class-name filtering.")
                print(f"[DEBUG] top_chunks assigned directly from raw_top_chunks.")
                print(f"[DEBUG] top_chunks count after TEMP skip = {len(top_chunks)}")
            # --------------------------------------------------
            # Step 4.4: Filter raw chunks by class_name
            # --------------------------------------------------
            #if fallback_reason is None:
             #   if class_list:
              ##     for c in raw_top_chunks:
                #        pdf_name = normalize_pdf_name(c.get("pdf_name", ""))
                 #       matched = matches_class(pdf_name, class_list)
######                   if matched:
      #                      filtered_chunks.append(c)
#
 #                   top_chunks = filtered_chunks
  #              else:
   #                 top_chunks = raw_top_chunks

    #            print(f"[DEBUG] top_chunks AFTER class filter: {len(top_chunks)}")

     #           if raw_top_chunks and not top_chunks:
      #              fallback_reason = "all_matches_removed_by_class_filter"
            print(f"[DEBUG] top_chunks BEFORE class filter: {len(top_chunks)}")
            print("\n================ TOP CHUNKS DEBUG ================")
            print(f"[DEBUG] Query: {query}")

            for i, c in enumerate(top_chunks, start=1):
                print(f"\n--- TOP CHUNK #{i} ---")
                print(f"pdf_name    = {c.get('pdf_name')}")
                print(f"page_number = {c.get('page_number')}")
                print(f"chunk_index = {c.get('chunk_index')}")
                print(f"pdf_link    = {c.get('pdf_link')}")
                print(f"score       = {c.get('score')}")
                chunk_text = c.get("chunk_text", "")
                print(f"chunk_text  = {chunk_text[:1500]}")
            print("==================================================\n")

            # --------------------------------------------------
            # TEMP DEBUG: Disable class-based filtering
            # --------------------------------------------------
            print("[DEBUG] TEMP: Skipping class-name filtering for FAISS chunks.")
            print(f"[DEBUG] class_list that would have been used = {class_list}")
            print(f"[DEBUG] top_chunks count after skipping class filter = {len(top_chunks)}")
            # --------------------------------------------------
            # Step 4.5: Sort final chunks
            # --------------------------------------------------
            if top_chunks:
                top_chunks = sorted(top_chunks, key=lambda x: x["score"], reverse=True)[:TOP_K]

                print("\n========== FILTERED TOP CHUNKS DEBUG ==========")
                for i, c in enumerate(top_chunks[:3], 1):
                    print(f"\n--- Filtered Chunk {i} ---")
                    for meta_key, meta_val in c.items():
                        print(f"{meta_key}: {meta_val}")
                print("===============================================\n")

        else:
            top_chunks = []
            raw_top_chunks = []
            fallback_reason = "faiss_unavailable"

    else:
        top_chunks = []
        raw_top_chunks = []
        if not pdf_files:
            fallback_reason = "no_pdf_files_for_context_search"
        elif use_context_only:
            fallback_reason = "use_context_only_enabled"

    # ------------------ Step 4.6: Decide whether GPT fallback is needed ------------------
    if not top_chunks:
        use_gpt_only = True

    print(f"[DEBUG] raw_top_chunks final count = {len(raw_top_chunks)}")
    print(f"[DEBUG] top_chunks final count     = {len(top_chunks)}")
    print(f"[DEBUG] use_gpt_only              = {use_gpt_only}")
    print(f"[DEBUG] fallback_reason           = {fallback_reason}")
    print("================ ACADEMY RETRIEVAL DEBUG END ================\n")

    # ------------------ Step 4.7: Build context string ------------------
    context_texts_str = "\n".join(
        f"PDF: {c.get('pdf_name', 'N/A')} (Page {c.get('page_number', 'N/A')})\n{c.get('chunk_text', '')}"
        for c in top_chunks
    )

    # ------------------ Step 5: Prepare GPT prompt ------------------
    reasoning_instruction = {
        "simple": (
            "Use plain, beginner-friendly language. Keep sentences short and avoid jargon. "
            "Provide only the final answer. Do NOT include step-by-step explanations, formulas, or LaTeX. "
            "If there is a number, include it with units, e.g., '112.5 ml'."
        ),
        "medium": (
            "Give a balanced explanation — clear, moderately detailed, and easy to follow. "
            "Provide only the final answer or main result. Avoid unnecessary formulas or LaTeX. "
            "Include numbers with units when applicable."
        ),
        "advanced": (
            "Provide a detailed, analytical, and example-rich explanation. "
            "Include the final answer clearly at the start. Minimize LaTeX or raw formulas. "
            "Numbers should be accompanied by units, e.g., '112.5 ml'."
        )
    }.get(reasoning, 
          "Use plain, beginner-friendly language. Provide only the final answer in plain text, without step-by-step explanation or LaTeX, include units if applicable."
    )


    #if use_context_only or not top_chunks:
     #   gpt_prompt = f"""


#You are an assistant. Follow the instructions below carefully.

#Style: {reasoning_instruction}

#Use only the previous conversation context to answer:
#{context_gist}

#Question:
#{query}

#Guidelines:
#- Do not use any external knowledge beyond the conversation.
#- If the user asks for a resource mentioned but not provided, say you do not have access.
##- Prepend "[GPT answer]" if relying on your own understanding.
#"""
    row = db.execute(select(FranchiseLocation)).scalar_one_or_none()

    country = ""
    state = ""

    if row:
        country = row.country or ""
        state = row.state or ""
        print(f"Country: {country}, State: {state}")
    else:
        print("No row found in FranchiseLocation table.")

    if use_gpt_only or not top_chunks:
        gpt_prompt = f"""
    You are an educational assistant for a {class_name} student in {student_year}, located in {country} {state}.

    Question:
    {query}

    Reasoning style:
    {reasoning_instruction}

    Instructions:
    - Answer in clean Markdown.
    - Use clean Markdown with compact spacing. Avoid unnecessary blank lines between short sections.
    - Write like a polished ChatGPT educational response.
    - Use short introductory context before the main answer when helpful.
    - Use headings, bullet points, numbered lists, and spacing where appropriate.
    - If the user asks for questions, worksheets, quizzes, explanations, summaries, or study help, structure the response neatly instead of writing one dense paragraph.
    - If the user asks for NSW Selective / OC / NAPLAN style material, match that exam style rather than giving generic school questions.
    - Always consider you are responding to school-going children.
    - Do not mention previous conversation unless explicitly asked.
    - Do not prepend labels like [GPT answer].
    - If you are unsure, say so clearly.

    Additional instruction for educational content:
    - If the user asks for multiple questions, format each question under its own numbered heading.
    - If the user asks for explanation, use sections and bullet points.
    - Avoid raw LaTeX unless necessary.
    """
    else:
        gpt_prompt = f"""
    You are an educational assistant for a {class_name} student in {student_year}, located in {country} {state}.

    Question:
    {query}

    Reasoning style:
    {reasoning_instruction}

    Relevant PDF chunks:
    {context_texts_str}

    Instructions:
    - Answer in clean Markdown.
    - Use clean Markdown with compact spacing. Avoid unnecessary blank lines between short sections.
    - Use the PDF chunks when they are relevant.
    - If the answer comes from the PDF content, base the answer on that material and do not invent facts.
    - Write like a polished ChatGPT educational response.
    - Use headings, bullets, numbering, and whitespace for readability.
    - Always consider you are responding to school-going children.
    - Do not prepend labels like [PDF-based answer] or [GPT answer].
    - If the PDF chunks are not enough, say so honestly rather than inventing.
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
        source_name = "Academy Answer" if top_chunks and not use_gpt_only else "GPT Answer"
    used_pdfs = []
    # ------------------ Step 8: Append PDF metadata + build PDF link ------------------
    top_pdf_name = None
    top_pdf_page = None
    top_pdf_file_id = None
    matched_pdf = None

    if source_name == "Academy Answer" and top_chunks:
        top_doc = top_chunks[0]

        # store PDF metadata for DB save later
        top_pdf_name = top_doc.get("pdf_name")
        top_pdf_page = top_doc.get("page_number")

        # append PDF metadata to assistant answer
        pdf_metadata = f"[PDF used: {top_pdf_name if top_pdf_name is not None else 'N/A'} (Page {top_pdf_page if top_pdf_page is not None else 'N/A'})]"
        answer_text = f"{answer_text}\n{pdf_metadata}"

        # build frontend PDF link for the top matched PDF
        normalized_top_pdf_name = normalize_pdf_name(top_pdf_name or "")
        top_page = top_pdf_page

        print("\n================ PDF LINK BUILD DEBUG START ================")
        print(f"[DEBUG] top_doc pdf_name   = {top_doc.get('pdf_name')}")
        print(f"[DEBUG] normalized name   = {normalized_top_pdf_name}")
        print(f"[DEBUG] top_doc page      = {top_page}")
        print(f"[DEBUG] real_student_id   = {real_student_id}")
        print(f"[DEBUG] allowed_pdf_files count = {len(allowed_pdf_files)}")

        for pdf in allowed_pdf_files:
            pdf_name_from_list = normalize_pdf_name(pdf.get("name", ""))
            print(f"[DEBUG] comparing against allowed pdf name = {pdf_name_from_list}")

            if pdf_name_from_list == normalized_top_pdf_name:
                matched_pdf = pdf
                break

        if matched_pdf:
            top_pdf_file_id = matched_pdf.get("id")

            frontend_url = (
                f"{FRONTEND_PUBLIC_BASE_URL}/pdf-viewer"
                f"?student_id={real_student_id}&file_id={top_pdf_file_id}"
            )

            if top_page:
                frontend_url += f"&page={top_page}"

            used_pdfs.append(frontend_url)

            print("[DEBUG] Matched PDF for academy source link")
            print(f"[DEBUG] matched_pdf name = {matched_pdf.get('name')}")
            print(f"[DEBUG] matched_pdf id   = {top_pdf_file_id}")
            print(f"[DEBUG] frontend_url     = {frontend_url}")
        else:
            print("[WARNING] Could not match top chunk pdf_name against allowed_pdf_files, so no PDF link will be attached.")

        print("================ PDF LINK BUILD DEBUG END ================\n")

    # ------------------ Step 9: Prepare final results ------------------
    
    # ------------------ Step 10: Update user context ------------------
    #append_to_user_context(user_id, "user", query)
    #append_to_user_context(user_id, "assistant", answer_text)
    final_links = used_pdfs if used_pdfs else []
    save_chatbot_message(
        db=db,
        conversation_id=conversation.id,
        role="assistant",
        message_text=answer_text,
        source_name=source_name,
        reasoning_level=reasoning,
        class_name=class_name or student.class_name,
        pdf_name=top_pdf_name,
        pdf_page=top_pdf_page,
        pdf_file_id=top_pdf_file_id,
        response_links=used_pdfs if source_name == "Academy Answer" else []
    )

    now_au = australia_now()

    conversation.last_message_at = now_au
    conversation.message_count = (conversation.message_count or 0) + 2
    conversation.updated_at = now_au

    db.commit()
    print("==================== SEARCH REQUEST END ====================\n")
    return JSONResponse({
        "source_name": source_name,
        "answer_markdown": answer_text,
        "links": final_links
    })


@app.post("/api/users/bulk")
async def upload_users(file: UploadFile = File(...), db: Session = Depends(get_db)):
    print("DEBUG: Bulk CSV upload request received")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. CSV required.")

    try:
        # Read CSV file safely
        df = pd.read_csv(
            file.file,
            sep=None,
            engine="python",
            dtype={"phone_number": str, "student_id": str},  # <-- support student_id
            encoding="utf-8-sig"
        )

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        print(f"DEBUG: Columns after normalization: {list(df.columns)}")

        # Required CSV fields
        required_columns = {"name", "email", "class_name", "class_day", "password", "student_id"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"CSV file must contain columns: {required_columns}"
            )

        # --- Normalize phone number format ---
        def fix_phone_number(x):
            if pd.isna(x):
                return None
            try:
                phone = str(int(float(x)))
            except:
                phone = str(x).strip().lstrip("'")
            phone = phone.replace(" ", "").replace("-", "")
            if phone.startswith("+"):
                return phone
            elif phone.startswith("04"):
                return f"+61{phone[1:]}"
            elif phone.startswith("4"):
                return f"+61{phone}"
            elif phone.startswith("61"):
                return f"+{phone}"
            else:
                return phone

        if "phone_number" in df.columns:
            df["phone_number"] = df["phone_number"].apply(fix_phone_number)

        # --- Fetch existing values to detect duplicates ---
        existing_users = db.query(User.email, User.phone_number, User.student_id).all()

        existing_emails = {u.email for u in existing_users if u.email}
        existing_phones = {u.phone_number for u in existing_users if u.phone_number}
        existing_student_ids = {u.student_id for u in existing_users if u.student_id}

        print(
            f"DEBUG: Existing emails={len(existing_emails)}, "
            f"phones={len(existing_phones)}, "
            f"student_ids={len(existing_student_ids)}"
        )

        users_to_add = []
        skipped_users = []

        # --- Process each row ---
        for index, row in df.iterrows():
            email = (row.get("email") or "").strip()
            phone = row.get("phone_number")
            student_id = (row.get("student_id") or "").strip()

            # Duplicate checks
            if (
                email in existing_emails
                or (phone and phone in existing_phones)
                or student_id in existing_student_ids
            ):
                skipped_users.append({
                    "name": row.get("name"),
                    "email": email,
                    "phone_number": phone,
                    "student_id": student_id,
                    "reason": "Duplicate email, phone number, or student_id"
                })
                print(f"DEBUG: Skipped duplicate -> {email} / {phone} / {student_id}")
                continue

            try:
                user_obj = User(
                    name=row["name"].strip(),
                    email=email,
                    phone_number=phone,
                    class_name=row.get("class_name"),
                    class_day=row.get("class_day"),
                    student_id=student_id,  # <-- NEW
                    password=row.get("password") or "placeholder",
                )

                users_to_add.append(user_obj)
                print(f"DEBUG: Prepared new user {index}: {email} / {student_id}")

            except Exception as e:
                skipped_users.append({
                    "name": row.get("name"),
                    "email": email,
                    "phone_number": phone,
                    "student_id": student_id,
                    "reason": f"Error processing row: {e}"
                })
                print(f"ERROR: Failed to process row {index}: {e}")
                continue

        if not users_to_add and not skipped_users:
            raise HTTPException(status_code=400, detail="No valid users found in CSV")

        # --- Insert new users ---
        if users_to_add:
            db.add_all(users_to_add)
            db.commit()
            print(f"DEBUG: Inserted {len(users_to_add)} users into DB")

        # --- Grant Google Drive access ---
        for u in users_to_add:
            try:
              give_drive_access(DEMO_FOLDER_ID, u.email, role="reader", db=db)
              print(f"DEBUG: Drive access granted -> {u.email}")
            except Exception as e:
              print(f"ERROR: Failed to grant Drive access: {u.email} -> {e}")

        # --- Response ---
        return {
            "added_users": [
                {
                    "name": u.name,
                    "email": u.email,
                    "phone_number": u.phone_number,
                    "class_name": u.class_name,
                    "class_day": u.class_day,
                    "student_id": u.student_id,  # <-- INCLUDED
                }
                for u in users_to_add
            ],
            "skipped_users": skipped_users,
            "summary": {
                "added": len(users_to_add),
                "skipped": len(skipped_users),
                "total_rows": len(df),
            }
        }

    except Exception as e:
        print(f"EXCEPTION: Bulk upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

"""
previous version of bulk upload without student_id
@app.post("/api/users/bulk")
async def upload_users(file: UploadFile = File(...), db: Session = Depends(get_db)):
    print("DEBUG: Bulk CSV upload request received")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. CSV required.")

    try:
        # Read CSV file safely
        df = pd.read_csv(
            file.file,
            sep=None,
            engine="python",
            dtype={"phone_number": str},
            encoding="utf-8-sig"
        )

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        print(f"DEBUG: Columns after normalization: {list(df.columns)}")

        # Ensure required columns exist
        required_columns = {"name", "email", "class_name", "class_day", "password"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"CSV file must contain columns: {required_columns}"
            )

        # --- Clean and normalize Australian phone numbers ---
        def fix_phone_number(x):
            if pd.isna(x):
                return None
            try:
                phone = str(int(float(x)))  # Handle scientific notation like 4.12E+09
            except:
                phone = str(x).strip().lstrip("'")
            phone = phone.replace(" ", "").replace("-", "")
            if phone.startswith("+"):
                return phone
            elif phone.startswith("04"):
                return f"+61{phone[1:]}"  # Convert 04... -> +61...
            elif phone.startswith("4"):
                return f"+61{phone}"  # Convert 4... -> +614...
            elif phone.startswith("61"):
                return f"+{phone}"  # Ensure leading +
            else:
                return phone  # fallback

        if "phone_number" in df.columns:
            df["phone_number"] = df["phone_number"].apply(fix_phone_number)

        # --- Fetch existing emails and phone numbers from DB ---
        existing_users = db.query(User.email, User.phone_number).all()
        existing_emails = {u.email for u in existing_users if u.email}
        existing_phones = {u.phone_number for u in existing_users if u.phone_number}
        print(f"DEBUG: Existing emails: {len(existing_emails)}, phones: {len(existing_phones)}")

        users_to_add = []
        skipped_users = []

        for index, row in df.iterrows():
            email = row.get("email", "").strip()
            phone = row.get("phone_number")

            # Skip duplicates
            if email in existing_emails or (phone and phone in existing_phones):
                skipped_users.append({
                    "name": row.get("name"),
                    "email": email,
                    "phone_number": phone,
                    "reason": "Duplicate email or phone number"
                })
                print(f"DEBUG: Skipped duplicate -> {email} / {phone}")
                continue

            try:
                user_obj = User(
                    name=row["name"].strip(),
                    email=email,
                    phone_number=phone,
                    class_name=row.get("class_name"),
                    class_day=row.get("class_day"),  # <-- added
                    password=row.get("password") or "placeholder",
                )
                users_to_add.append(user_obj)
                print(f"DEBUG: Prepared new user {index}: {email}")
            except Exception as e:
                skipped_users.append({
                    "name": row.get("name"),
                    "email": email,
                    "phone_number": phone,
                    "reason": f"Error processing row: {e}"
                })
                print(f"ERROR: Failed to process row {index}: {e}")
                continue

        if not users_to_add and not skipped_users:
            raise HTTPException(status_code=400, detail="No valid users found in CSV")

        # --- Insert new users ---
        if users_to_add:
            db.add_all(users_to_add)
            db.commit()
            print(f"DEBUG: Inserted {len(users_to_add)} users into the database")

        # --- Grant Google Drive access ---
        for u in users_to_add:
            try:
                give_drive_access(DEMO_FOLDER_ID, u.email, role="reader", db=db)
                print(f"DEBUG: Granted Drive access to {u.email}")
            except Exception as e:
                print(f"ERROR: Failed to give Drive access to {u.email}: {e}")

        # --- Return detailed response ---
        return {
            "added_users": [
                {
                    "name": u.name,
                    "email": u.email,
                    "phone_number": u.phone_number,
                    "class_name": u.class_name,
                    "class_day": u.class_day,
                }
                for u in users_to_add
            ],
            "skipped_users": skipped_users,
            "summary": {
                "added": len(users_to_add),
                "skipped": len(skipped_users),
                "total_rows": len(df),
            }
        }

    except Exception as e:
        print(f"EXCEPTION: Bulk upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 """


# Utility: hash password

# Bulk upload endpoint
def load_vectorstore_from_gcs_in_memory(gcs_prefix: str, embeddings: OpenAIEmbeddings) -> FAISS:
    """
    Load a FAISS vector store from GCS directly into memory.
    Handles docstore tuples or dict/InMemoryDocstore.
    """
    gcs_prefix = gcs_prefix.replace("\\", "/")

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

def normalize_pdf_name(name: str) -> str:
    return name.lower().replace(".pdf.pdf", ".pdf").replace("- ", "-").strip()

from time import perf_counter
from sqlalchemy.orm import load_only

@app.post("/admin/initialize_faiss")
def initialize_faiss(db: Session = Depends(get_db)):
    """
    Stream embeddings from DB, rebuild FAISS in memory, and log progress so we
    can tell whether it is genuinely working slowly or actually stuck.
    """
    global FAISS_INDEX, FAISS_METADATA

    print("\n================ INITIALIZE FAISS START ================", flush=True)
    t0 = perf_counter()

    try:
        # --------------------------------------------------
        # Step 1: Count rows first
        # --------------------------------------------------
        print("[STEP 1] Counting embeddings in DB...", flush=True)
        count_start = perf_counter()
        total_embeddings = db.query(Embedding).count()
        print(
            f"[STEP 1 DONE] Total embeddings in DB: {total_embeddings} "
            f"(took {perf_counter() - count_start:.2f}s)",
            flush=True
        )

        if total_embeddings == 0:
            raise HTTPException(status_code=404, detail="No embeddings found in DB.")

        # --------------------------------------------------
        # Step 2: Stream only needed columns in batches
        # --------------------------------------------------
        print("[STEP 2] Streaming embeddings from DB in batches...", flush=True)
        stream_start = perf_counter()

        query = (
            db.query(Embedding)
            .options(
                load_only(
                    Embedding.pdf_name,
                    Embedding.class_name,
                    Embedding.page_number,
                    Embedding.chunk_index,
                    Embedding.pdf_link,
                    Embedding.chunk_text,
                    Embedding.embedding_vector,
                )
            )
            .yield_per(200)
        )

        vectors_list = []
        metadata_list = []
        processed = 0

        for e in query:
            vectors_list.append(e.embedding_vector)
            metadata_list.append({
                "pdf_name": normalize_pdf_name(e.pdf_name),
                "class_name": e.class_name,
                "page_number": e.page_number,
                "chunk_index": e.chunk_index,
                "pdf_link": e.pdf_link,
                "chunk_text": e.chunk_text
            })

            processed += 1

            # Show the first row shape/details once for sanity
            if processed == 1:
                first_vec_len = len(e.embedding_vector) if e.embedding_vector is not None else None
                print("[STEP 2 SAMPLE] First embedding row:", flush=True)
                print(f"    pdf_name      = {e.pdf_name}", flush=True)
                print(f"    class_name    = {e.class_name}", flush=True)
                print(f"    page_number   = {e.page_number}", flush=True)
                print(f"    chunk_index   = {e.chunk_index}", flush=True)
                print(f"    vector_type   = {type(e.embedding_vector)}", flush=True)
                print(f"    vector_length = {first_vec_len}", flush=True)

            if processed % 200 == 0:
                print(
                    f"[STEP 2 PROGRESS] Processed {processed}/{total_embeddings} embeddings "
                    f"({perf_counter() - stream_start:.2f}s elapsed)",
                    flush=True
                )

        print(
            f"[STEP 2 DONE] Finished streaming {processed} embeddings "
            f"(took {perf_counter() - stream_start:.2f}s)",
            flush=True
        )

        if processed != total_embeddings:
            print(
                f"[WARN] Count mismatch: count()={total_embeddings}, streamed={processed}",
                flush=True
            )

        # --------------------------------------------------
        # Step 3: Convert to NumPy
        # --------------------------------------------------
        print("[STEP 3] Converting vectors to NumPy array...", flush=True)
        np_start = perf_counter()

        vectors = np.array(vectors_list, dtype="float32")

        print(
            f"[STEP 3 DONE] NumPy array created in {perf_counter() - np_start:.2f}s",
            flush=True
        )
        print(f"    vectors.shape = {vectors.shape}", flush=True)
        print(f"    vectors.dtype = {vectors.dtype}", flush=True)

        if len(vectors.shape) != 2:
            raise ValueError(
                f"Embedding vectors array has invalid shape {vectors.shape}. "
                f"Expected 2D array like (num_embeddings, embedding_dim)."
            )

        if vectors.shape[0] == 0:
            raise ValueError("No vectors available after conversion.")

        # --------------------------------------------------
        # Step 4: Create FAISS index
        # --------------------------------------------------
        d = vectors.shape[1]
        print(f"[STEP 4] Creating FAISS index with dimension d={d}...", flush=True)
        faiss_create_start = perf_counter()

        temp_index = faiss.IndexFlatIP(d)

        print(
            f"[STEP 4 DONE] FAISS index object created in "
            f"{perf_counter() - faiss_create_start:.2f}s",
            flush=True
        )

        # --------------------------------------------------
        # Step 5: Normalize vectors
        # --------------------------------------------------
        print("[STEP 5] Normalizing vectors for cosine similarity...", flush=True)
        norm_start = perf_counter()

        faiss.normalize_L2(vectors)

        print(
            f"[STEP 5 DONE] Vector normalization complete in "
            f"{perf_counter() - norm_start:.2f}s",
            flush=True
        )

        # --------------------------------------------------
        # Step 6: Add vectors to FAISS
        # --------------------------------------------------
        print("[STEP 6] Adding vectors to FAISS index...", flush=True)
        add_start = perf_counter()

        temp_index.add(vectors)

        print(
            f"[STEP 6 DONE] Added {temp_index.ntotal} vectors to FAISS "
            f"in {perf_counter() - add_start:.2f}s",
            flush=True
        )

        # --------------------------------------------------
        # Step 7: Swap globals only after success
        # --------------------------------------------------
        print("[STEP 7] Updating in-memory globals...", flush=True)
        FAISS_INDEX = temp_index
        FAISS_METADATA = metadata_list
        print(
            f"[STEP 7 DONE] FAISS_INDEX and FAISS_METADATA updated "
            f"({len(FAISS_METADATA)} metadata rows)",
            flush=True
        )

        total_time = perf_counter() - t0
        print("[SUCCESS] FAISS index initialized successfully.", flush=True)
        print(f"          Total embeddings indexed: {temp_index.ntotal}", flush=True)
        print(f"          Total time: {total_time:.2f}s", flush=True)
        print("================ INITIALIZE FAISS END ================\n", flush=True)

        return {
            "message": "FAISS index initialized successfully",
            "num_embeddings": temp_index.ntotal,
            "dimension": d,
            "total_time_seconds": round(total_time, 2),
        }

    except HTTPException as http_err:
        print(f"[HTTP ERROR] {http_err.detail}", flush=True)
        print("================ INITIALIZE FAISS FAILED ================\n", flush=True)
        raise http_err

    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to initialize FAISS index: {str(e)}", flush=True)
        traceback.print_exc()
        print("================ INITIALIZE FAISS FAILED ================\n", flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize FAISS index: {str(e)}"
        )

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
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )

        # Step 1: Process vector stores
        for pdf_idx, pdf in enumerate(all_pdfs, start=1):
            pdf_name = pdf.get("name")
            pdf_path = pdf.get("path")
            pdf_base_name = pdf_name.rsplit(".", 1)[0]

            parent_folder = pdf_path.rsplit("/", 1)[0] if "/" in pdf_path else ""
            gcs_prefix = f"{parent_folder}/vectorstore_{pdf_base_name}/".replace("\\", "/")

            print(f"\n[DEBUG] Processing PDF {pdf_idx}/{len(all_pdfs)}: {pdf_name}")
            print(f"[DEBUG] Loading vector store from GCS: {gcs_prefix}")

            try:
                prefix_blobs = list(gcs_bucket.list_blobs(prefix=gcs_prefix, max_results=10))
                print(f"[DEBUG] blob count under prefix = {len(prefix_blobs)}")
                for b in prefix_blobs:
                    print(f"    [PREFIX BLOB] {b.name}")
            except Exception as e:
                print(f"[ERROR] Failed listing prefix blobs: {e}")

            print("============================================================\n")

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

