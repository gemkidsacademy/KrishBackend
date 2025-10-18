import random
import time
from dotenv import load_dotenv


#Twilio API
from twilio.rest import Client
# FastAPI & Pydantic
from fastapi import FastAPI, Response, Depends, HTTPException, Query
from pydantic import BaseModel
from passlib.hash import pbkdf2_sha256

# SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from sqlalchemy.dialects.postgresql import UUID

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



#-------------------------------- for Twilio
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
client = Client(account_sid, auth_token)
# Temporary in-memory OTP storage
otp_store = {}


# -----------------------------
# App & CORS
# -----------------------------
load_dotenv()
app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://krish-chat-bot.vercel.app",  # your Vercel frontend
        "http://localhost:3000"               # optional for local dev
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

DEMO_FOLDER_ID = "1lyKKM94QxpLf0Re76_1rGuk5gCRWcuP0"

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
gcs_bucket_name = "krishdemochatbot"
gcs_bucket = gcs_client.bucket(gcs_bucket_name)

print(f"✅ Initialized GCS client for bucket: {gcs_bucket_name}")

#---------------- database connectivity 
DATABASE_URL = os.environ.get("DATABASE_URL") or (
    f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class UsageResponse(BaseModel):
    date: str
    amount_usd: float
    type: str

class SendOTPRequest(BaseModel):
    phone_number: str

class VerifyOTPRequest(BaseModel):
    phone_number: str
    otp: str

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

class AddUserRequest(BaseModel):
    id: int
    name: str
    email: str
    password: str
    phone_number: str | None = None
    class_name: str | None = None

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

# Send OTP SMS
def send_otp_sms(phone_number: str, otp: int):
    message = client.messages.create(
        body=f"Your OTP code is {otp}",
        from_=twilio_number,
        to=phone_number
    )
    print(f"Sent OTP {otp} to {phone_number}, SID: {message.sid}")
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

@app.get("/api/usage", response_model=list[UsageResponse])
def get_openai_usage(days: int = 30):
    """
    Retrieve OpenAI API usage for the past `days` days.
    """
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.utcnow().strftime("%Y-%m-%d")

    usage_data = openai_client.usage.list(
        start_date=start_date,
        end_date=end_date
    )

    response = [
        UsageResponse(
            date=item["date"],
            amount_usd=item["amount"] / 100,  # convert cents to dollars
            type=item["type"]
        )
        for item in usage_data.data
    ]

    return response

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
    print(f"[DEBUG] Received OTP verification request for phone number: {data.phone_number}")

    # Retrieve OTP record from memory
    record = otp_store.get(data.phone_number)
    if not record:
        print(f"[WARNING] No OTP record found for {data.phone_number}")
        raise HTTPException(status_code=400, detail="OTP not sent")

    if time.time() > record["expiry"]:
        print(f"[WARNING] OTP for {data.phone_number} has expired")
        raise HTTPException(status_code=400, detail="OTP expired")

    print(f"[DEBUG] Comparing entered OTP '{data.otp}' with stored OTP '{record['otp']}'")
    if str(data.otp) != str(record["otp"]):
        print(f"[WARNING] Entered OTP ({data.otp}) does not match stored OTP ({record['otp']})")
        raise HTTPException(status_code=400, detail="Invalid OTP")

    # OTP is valid
    print(f"[INFO] OTP for {data.phone_number} is valid")

    # Fetch user info from database
    user = db.query(User).filter(User.phone_number == data.phone_number).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_info = {
        "id": user.id,
        "name": user.name,
        "phone_number": user.phone_number,
        # optionally add session_token or other fields if needed
    }

    # Clear OTP from memory after successful verification
    del otp_store[data.phone_number]
    print(f"[DEBUG] Cleared OTP for {data.phone_number} after successful verification")

    return {"message": "OTP verified successfully", "user": user_info}

@app.post("/login")
async def login(
    login_request: LoginRequest,
    response: Response,
    db: Session = Depends(get_db)
):
    print("\n==================== LOGIN ATTEMPT START ====================")
    global user_contexts

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
    
        print("DEBUG: Stored hash in DB:", user.password)
        print("DEBUG: Password provided by user:", password)
    
        password_verified = check_password_hash(user.password, password)
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
        id=user_request.id,
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

    return {"message": f"User '{new_user.name}' added successfully!"}

        

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
    Includes detailed debug print statements.
    """
    results = []
    page_token = None

    print(f"DEBUG: Starting to list files in folder_id='{folder_id}' with path='{path}'")

    while True:
        response = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType, webViewLink)',
            pageToken=page_token
        ).execute()

        files = response.get('files', [])
        print(f"DEBUG: Retrieved {len(files)} files from folder_id='{folder_id}'")

        for file in files:
            file_id = file['id']
            file_name = file['name']
            mime_type = file['mimeType']
            web_view_link = file.get('webViewLink', '')

            if mime_type == 'application/pdf':
                pdf_path = f"{path}/{file_name}".lstrip("/")
                results.append({
                    "id": file_id,
                    "name": file_name,
                    "webViewLink": web_view_link,
                    "path": pdf_path
                })
                print(f"DEBUG: Found PDF -> id: {file_id}, name: '{file_name}', path: '{pdf_path}'")

            elif mime_type == 'application/vnd.google-apps.folder':
                folder_path = f"{path}/{file_name}".lstrip("/")
                print(f"DEBUG: Found folder -> id: {file_id}, name: '{file_name}', path: '{folder_path}'")
                # Recursively list PDFs inside this folder
                nested_results = list_pdfs(file_id, folder_path)
                results.extend(nested_results)
                print(f"DEBUG: Completed folder '{folder_path}', found {len(nested_results)} PDFs inside")

            else:
                print(f"DEBUG: Skipping file -> id: {file_id}, name: '{file_name}', mimeType: '{mime_type}'")

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
    Process a single PDF from Google Drive and create a separate vector store for it in Google Cloud Storage (GCS).

    This function ensures:
    1. Each PDF gets its own vector store, preserving Google Drive folder structure in GCS.
    2. PDFs already uploaded and processed are skipped to save time.
    3. Embeddings are created using OpenAI's 'text-embedding-3-large' model.
    4. Vector store files are uploaded recursively to GCS.
    5. Original PDF is also uploaded to GCS.
    
    Note:
    - This does NOT merge PDFs in a folder; each PDF is handled individually.
    - If a new PDF is added to an existing folder, it will be processed and a new vector store created.
    """

    pdf_name = pdf_file.get("name", "Unknown")
    pdf_id = pdf_file.get("id")
    pdf_path = pdf_file.get("path", pdf_name)  # preserve folder structure
    pdf_link = pdf_file.get("webViewLink", "")

    print(f"[DEBUG] Starting vector store creation for PDF: {pdf_name}")

    # Skip PDFs with no Google Drive ID
    if pdf_id is None:
        print(f"[DEBUG] PDF {pdf_name} has no Drive ID. Skipping.")
        return

    # Skip PDFs that already exist in GCS
    if gcs_bucket.blob(pdf_path).exists():
        print(f"[DEBUG] PDF {pdf_name} already exists in GCS. Skipping upload.")
        return

    # Step 1: Download PDF from Drive
    try:
        print(f"[DEBUG] Downloading PDF: {pdf_name}")
        file_bytes = download_pdf(pdf_id)
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        print(f"[ERROR] Failed to download or read PDF {pdf_name}: {e}")
        return

    # Step 2: Split PDF into text chunks
    chunks = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                cleaned_text = " ".join(line.strip() for line in page_text.splitlines() if line.strip())
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=250,
                    separators=["\n\n", "\n", " "]
                )
                page_chunks = text_splitter.create_documents([cleaned_text])

                # Add metadata to each chunk
                for j, chunk in enumerate(page_chunks, start=1):
                    chunk.metadata.update({
                        "pdf_name": pdf_name,
                        "page_number": i,
                        "chunk_index": j,
                        "pdf_link": pdf_link
                    })
                    chunks.append(chunk)
        except Exception as e:
            print(f"[ERROR] Failed to process page {i} of PDF {pdf_name}: {e}")

    if not chunks:
        print(f"[DEBUG] No text found in PDF {pdf_name}, skipping vector store creation.")
        return

    # Step 3: Create embeddings and FAISS vector store
    try:
        print(f"[DEBUG] Creating embeddings for PDF: {pdf_name}")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.environ.get("OPENAI_API_KEY_S")
        )
        vs = FAISS.from_documents(chunks, embeddings)
        if hasattr(vs.index, "normalize_L2"):
            vs.index.normalize_L2()  # optional: make cosine similarity scores accurate
        print(f"[DEBUG] Vector store created for PDF: {pdf_name}")
    except Exception as e:
        print(f"[ERROR] Failed to create embeddings/vector store for PDF {pdf_name}: {e}")
        return

    # Step 4: Save vector store recursively to GCS
    try:
        gcs_prefix_vs = f"{os.path.dirname(pdf_path)}/vectorstore/"
        with tempfile.TemporaryDirectory() as tmp_dir:
            vs.save_local(tmp_dir)

            # Recursively upload all vector store files
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
    # Initialize Google Cloud Storage client
    

    # List all blobs under the given prefix
    blobs = list(gcs_client.list_blobs(gcs_bucket_name, prefix=gcs_prefix))
    if not blobs:
        raise ValueError(f"No vector store files found in GCS for prefix '{gcs_prefix}'.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download each blob to the temp directory
        for blob in blobs:
            file_path = os.path.join(tmp_dir, os.path.basename(blob.name))
            blob.download_to_filename(file_path)

        # Load the vector store from the temp directory
        vs = FAISS.load_local(
            tmp_dir,
            embeddings,
            allow_dangerous_deserialization=True  # Safe for your own files
        )

    return vs
    
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

def classify_query_type(query: str, context_gist: str) -> str:
    """Use GPT to classify if query is context_only, pdf_only, or mixed."""
    prompt = f"""
    You are a classifier assistant. 
    Determine whether the following user query requires:
    1. Only the previous conversation context to answer (context_only)
    2. Only new information from PDF resources (pdf_only)
    3. Both previous context and PDFs (mixed)

    Previous Context (gist):
    {context_gist}

    User Query:
    {query}

    Respond with exactly one word: context_only, pdf_only, or mixed.
    """
    response = openai_client.chat.completions.create(
        model=REWRITER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

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


vectorstores_initialized = False

@app.get("/search")
async def search_pdfs(
    query: str = Query(..., min_length=1),
    reasoning: str = Query("simple", regex="^(simple|medium|advanced)$"),
    user_id: str = Query(...),
    class_name: str = Query(None)  # optional, for folder filtering
):
    print(f"\n==================== SEARCH REQUEST START ====================")
    print(f"user_id: {user_id}, query: {query}, reasoning: {reasoning}, class_name: {class_name}")

    global vectorstores_initialized
    if user_id not in user_contexts:
        user_contexts[user_id] = []

    results, top_chunks = [], []

    # -------------------- Step 0: Context gist --------------------
    context_gist = get_context_gist(user_id)
    is_first_query = len(user_contexts[user_id]) == 0
    query_type = classify_query_type(query, context_gist)
    if is_first_query:
        query_type = "pdf_only"

    use_context_only = query_type == "context_only"

    # -------------------- Step 1: Retrieve PDFs --------------------
    pdf_files = []
    if query_type in ("pdf_only", "mixed") and class_name:
        all_pdfs = list_pdfs(DEMO_FOLDER_ID)
        pdf_files = [pdf for pdf in all_pdfs if pdf.get("path", "").lower().startswith(class_name.lower())]
        print(f"[DEBUG] PDFs matching class '{class_name}': {len(pdf_files)}")

        for pdf in pdf_files:
            print(f"[DEBUG]   {pdf['name']} | Path: {pdf['path']}")

        if pdf_files and not vectorstores_initialized:
            ensure_vectorstores_for_all_pdfs(pdf_files)
            vectorstores_initialized = True

        if not pdf_files:
            print(f"[WARNING] No PDFs found for '{class_name}'. Falling back to GPT only.")
            use_context_only = True

    # -------------------- Step 2: Retrieve relevant PDF chunks --------------------
    context_texts_str = ""
    if pdf_files and not use_context_only:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.environ.get("OPENAI_API_KEY_S"))
        rewritten_query = query
        for pdf in pdf_files:
            pdf_name = pdf["name"]
            pdf_base_name = pdf_name.rsplit(".", 1)[0]
            gcs_prefix = os.path.join(os.path.dirname(pdf["path"]), "vectorstore") + "/"
            vectorstore: FAISS = load_vectorstore_from_gcs(gcs_prefix, embeddings)
            if hasattr(vectorstore, "index") and hasattr(vectorstore.index, "normalize_L2"):
                vectorstore.index.normalize_L2()
            docs_with_scores = vectorstore.similarity_search_with_score(rewritten_query, k=TOP_K)
            for doc, score in docs_with_scores:
                doc.metadata["pdf_name"] = pdf_name
                doc.metadata["pdf_base_name"] = pdf_base_name
                top_chunks.append((doc, score))

        top_chunks = sorted(top_chunks, key=lambda x: x[1])[:TOP_K]
        if top_chunks:
            context_texts = [
                f"PDF: {doc.metadata['pdf_name']} (Page {doc.metadata.get('page_number', 'N/A')})\n{doc.page_content}"
                for doc, _ in top_chunks
            ]
            context_texts_str = "\n".join(context_texts)

    # -------------------- Step 3: Prepare GPT prompt --------------------
    reasoning_instruction = {
        "simple": "Answer concisely in simple language.",
        "medium": "Answer with moderate detail.",
        "advanced": "Provide an in-depth, analytical answer."
    }.get(reasoning, "Answer concisely in simple language.")

    if use_context_only or not top_chunks:
        gpt_prompt = f"""
        Use only the previous conversation context to answer:
        {context_gist}

        Question: {query}

        Answer concisely. Prepend "[GPT answer]" if relying on own knowledge.
        """
    else:
        gpt_prompt = f"""
        You are an assistant. Use the following to answer:
        Previous conversation context:
        {context_gist}

        PDF Chunks:
        {context_texts_str}

        Question: {query}

        Instructions:
        1. Use PDF chunks if relevant.
        2. Do not invent facts.
        3. Prepend "[PDF-based answer]" if using PDFs, else "[GPT answer]".
        4. Style: {reasoning_instruction}
        """

    print("[DEBUG] GPT PROMPT PREVIEW:", gpt_prompt[:800])

    # -------------------- Step 4: Call GPT --------------------
    answer_response = openai_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": gpt_prompt}],
        temperature=0.2
    )

    answer_text = answer_response.choices[0].message.content.strip()
    print("[DEBUG] GPT RAW RESPONSE:", answer_text[:500])

    # -------------------- Step 5: Determine Source --------------------
    if answer_text.startswith("[PDF-based answer]"):
        source_name = "Academy Answer"
        answer_text = answer_text.replace("[PDF-based answer]", "", 1).strip()
    elif answer_text.startswith("[GPT answer]"):
        source_name = "GPT Answer"
        answer_text = answer_text.replace("[GPT answer]", "", 1).strip()
    else:
        # If PDFs existed but model didn’t label response, infer by checking relevance
        if top_chunks and any(word in answer_text.lower() for word in ["page", "pdf", "worksheet"]):
            source_name = "Academy Answer"
        else:
            source_name = "GPT Answer"

    # -------------------- Step 6: Add context if GPT used --------------------
    if source_name == "GPT Answer":
        answer_text = (
            "The answer was not found in the available PDFs, so GPT is using its own external knowledge base. "
            + answer_text
        )

    # -------------------- Step 7: Collect PDF links --------------------
    used_pdfs = list({doc.metadata.get("pdf_link") for doc, _ in top_chunks if doc.metadata.get("pdf_link")})

    # -------------------- Step 8: Append PDF metadata if Academy Answer --------------------
    if source_name == "Academy Answer" and top_chunks:
        top_doc, _ = top_chunks[0]
        pdf_metadata = f"[PDF used: {top_doc.metadata['pdf_name']} (Page {top_doc.metadata.get('page_number','N/A')})]"
        answer_text = f"{answer_text}\n{pdf_metadata}"

    # -------------------- Step 9: Prepare results --------------------
    results.append({
        "name": f"**{source_name}**",
        "snippet": answer_text,
        "links": used_pdfs if source_name == "Academy Answer" else []
    })

    # -------------------- Step 10: Update context --------------------
    append_to_user_context(user_id, "user", query)
    append_to_user_context(user_id, "assistant", answer_text)

    print(f"[INFO] Source detected: {source_name}")
    print("==================== SEARCH REQUEST END ====================\n")

    return JSONResponse(results)














# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo_chatbot_backend_2:app", host="0.0.0.0", port=8000, reload=True)
