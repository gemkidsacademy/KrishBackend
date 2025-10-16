# FastAPI & Pydantic
from fastapi import FastAPI, Response, Depends, HTTPException
from pydantic import BaseModel

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
from datetime import datetime
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
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Rapidfuzz for string matching
from rapidfuzz import fuzz
# -----------------------------
# App & CORS
# -----------------------------
app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://krish-chat-bot.vercel.app",
        "https://your-other-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
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
service_account_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])

# Initialize GCS client and get the bucket
gcs_client = storage.Client(credentials=Credentials.from_service_account_info(service_account_info),
                            project=service_account_info["project_id"])
gcs_bucket_name = "krishdemochatbot"
gcs_bucket = gcs_client.bucket(gcs_bucket_name)


#---------------- database connectivity 
DATABASE_URL = os.environ.get("DATABASE_URL") or (
    f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

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
    username: str
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
        
# -----------------------------
# API Endpoints
# -----------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@app.post("/login")
async def login(
    login_request: LoginRequest,
    response: Response,
    db: Session = Depends(get_db)
):
    print("\n==================== LOGIN ATTEMPT START ====================")

    # Log the incoming request data
    try:
        print("DEBUG: Raw login_request object:", login_request)
        print("DEBUG: login_request type:", type(login_request))
        print("DEBUG: login_request.dict():", login_request.dict())
        print(f"DEBUG: Username received: {login_request.username}")
        print(f"DEBUG: Password received: {login_request.password}")
    except Exception as e:
        print("ERROR: Failed to print login_request details:", e)

    # Fetch user from DB
    print("\n--- Step 1: Fetching user from DB ---")
    try:
        user = db.query(User).filter(User.username == login_request.username).first()
        print("DEBUG: Raw user fetched:", user)
        if user:
            print(f"DEBUG: User found: ID={user.id}, Username={user.username}")
            print(f"DEBUG: Stored (hashed) password: {user.password}")
        else:
            print("DEBUG: No user found in DB with that username.")
    except Exception as e:
        print("ERROR while querying user:", e)
        raise HTTPException(status_code=500, detail="Database error")

    if not user:
        print("DEBUG: Aborting - invalid username.")
        print("==================== LOGIN ATTEMPT END ====================\n")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Step 2: Password Verification
    print("\n--- Step 2: Verifying password ---")
    try:
        if login_request.password is None:
            print("DEBUG: Password missing from request.")
        elif user.password is None:
            print("DEBUG: Stored password is None in DB.")

        print("DEBUG: Comparing provided password vs stored hash...")
        password_verified = pwd_context.verify(login_request.password, user.password)
        print("DEBUG: pwd_context.verify result:", password_verified)
    except Exception as e:
        print("ERROR during password verification:", e)
        raise HTTPException(status_code=500, detail="Password verification error")

    if not password_verified:
        print(f"DEBUG: Invalid password for user {login_request.username}")
        print("DEBUG: Provided password:", login_request.password)
        print("DEBUG: Expected (hashed) password:", user.password)
        print("==================== LOGIN ATTEMPT END ====================\n")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    print("DEBUG: Password verified successfully.")

    # Step 3: Check for existing session
    print("\n--- Step 3: Checking existing session ---")
    try:
        existing_session = db.query(SessionModel).filter(SessionModel.user_id == user.id).first()
        print("DEBUG: Existing session query result:", existing_session)
    except Exception as e:
        print("ERROR while checking existing session:", e)
        existing_session = None

    if existing_session:
        session_token = existing_session.session_token
        public_token = existing_session.public_token
        print(f"DEBUG: Reusing existing session for user {user.id}")
        print(f"DEBUG: session_token={session_token}")
        print(f"DEBUG: public_token={public_token}")
    else:
        print("DEBUG: No existing session found. Creating new session...")
        try:
            session_token = str(uuid4())
            public_token = str(uuid4())
            print(f"DEBUG: Generated session_token={session_token}")
            print(f"DEBUG: Generated public_token={public_token}")

            new_session = SessionModel(
                session_token=session_token,
                user_id=user.id,
                public_token=public_token
            )
            db.add(new_session)
            db.commit()
            print("DEBUG: New session stored successfully.")
        except Exception as e:
            print("ERROR while creating new session:", e)
            raise HTTPException(status_code=500, detail="Session creation failed")

    # Step 4: Set session cookie
    print("\n--- Step 4: Setting cookie ---")
    try:
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            secure=False,  # Change to True if using HTTPS
            samesite="Lax",
            max_age=3600
        )
        print("DEBUG: Cookie set successfully with session_token:", session_token)
    except Exception as e:
        print("ERROR while setting cookie:", e)

    # Step 5: Prepare and return response
    print("\n--- Step 5: Preparing response ---")
    response_content = {
        "message": "Login successful",
        "id": user.id,
        "name": user.name,
        "email": getattr(user, "email", None),
        "session_token": session_token,
        "public_token": public_token
    }
    print("DEBUG: Response content prepared:", response_content)

    print("==================== LOGIN ATTEMPT SUCCESS ====================\n")

    return JSONResponse(
        content=response_content,
        status_code=200
    )


        

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

def list_pdfs(folder_id, level=0):
    results = []
    indent = "  " * level
    page_token = None

    while True:
        response = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType, webViewLink)',
            pageToken=page_token
        ).execute()

        for file in response.get('files', []):
            if file['mimeType'] == 'application/pdf':
                results.append({
                    "id": file['id'],
                    "name": file['name'],
                    "webViewLink": file.get('webViewLink', ''),
                    "parent_id": folder_id
                })
            elif file['mimeType'] == 'application/vnd.google-apps.folder':
                results.extend(list_pdfs(file['id'], level + 1))

        page_token = response.get('nextPageToken', None)
        if not page_token:
            break

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
    Create a vector store for a single PDF and upload it to the correct GCS bucket.
    Each chunk includes metadata: page number, chunk index, and Google Drive link.
    Skips processing if the PDF already exists in the bucket.
    Embeddings are normalized so FAISS similarity scores are in [0, 1] (cosine similarity).
    """

    pdf_name = pdf_file.get("name")
    pdf_base_name = pdf_name.replace(".pdf", "")
    pdf_link = pdf_file.get("webViewLink", "")

    # Check if PDF already exists in GCS
    pdf_blob_name = f"{pdf_base_name}/{pdf_name}"
    if gcs_bucket.blob(pdf_blob_name).exists():
        print(f"[DEBUG] PDF {pdf_name} already exists in GCS. Skipping upload.")
        return

    # Download PDF bytes from Drive
    pdf_id = pdf_file.get("id")
    if pdf_id is None:
        print(f"[DEBUG] PDF {pdf_name} has no Drive ID. Skipping.")
        return
    file_bytes = download_pdf(pdf_id)
    reader = PdfReader(io.BytesIO(file_bytes))

    # Prepare chunks with metadata
    chunks = []
    print(f"[DEBUG] Processing PDF: {pdf_name}, Total pages: {len(reader.pages)}")
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            cleaned_text = " ".join(
                line.strip() for line in page_text.splitlines() if line.strip()
            )

            # Semantic chunking: larger chunk size with overlap
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=250,
                separators=["\n\n", "\n", " "]
            )
            page_chunks = text_splitter.create_documents([cleaned_text])

            for j, chunk in enumerate(page_chunks, start=1):
                chunk.metadata.update({
                    "pdf_name": pdf_name,
                    "pdf_base_name": pdf_base_name,
                    "page_number": i,
                    "chunk_index": j,
                    "pdf_link": pdf_link,
                    "chunk_length": len(chunk.page_content)
                })
                chunks.append(chunk)
                print(f"[DEBUG] Chunk added: PDF={pdf_name}, Page={i}, Chunk={j}, Text sample='{chunk.page_content[:100]}...'")

    if not chunks:
        print(f"[DEBUG] No text found in PDF {pdf_name}, skipping vector store creation.")
        return

    # Create embeddings using a strong model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.environ.get("OPENAI_API_KEY_S")
    )

    # Create FAISS vector store
    vs = FAISS.from_documents(chunks, embeddings)
    print(f"[DEBUG] FAISS vector store created for PDF {pdf_name}, Total chunks: {len(chunks)}")

    # 🔹 Normalize embeddings so FAISS returns cosine similarity in [0,1]
    if hasattr(vs, "index") and hasattr(vs.index, "normalize_L2"):
        vs.index.normalize_L2()
        print("[DEBUG] FAISS embeddings normalized (unit vectors) for cosine similarity.")

    # Inspect embedding vectors for a few chunks
    for idx, doc in enumerate(chunks[:5]):
        vec = embeddings.embed_query(doc.page_content)
        norm = sum([v**2 for v in vec])**0.5
        print(f"[DEBUG] Embedding vector sample for chunk {idx+1}: Norm={norm:.4f}, Text sample='{doc.page_content[:80]}...'")

    # Save locally and upload vector store files to GCS
    gcs_prefix_vs = f"{pdf_base_name}/vectorstore/"
    with tempfile.TemporaryDirectory() as tmp_dir:
        vs.save_local(tmp_dir)
        for filename in os.listdir(tmp_dir):
            path = os.path.join(tmp_dir, filename)
            with open(path, "rb") as f:
                blob_name = f"{gcs_prefix_vs}{filename}"
                upload_to_gcs(f.read(), blob_name)
                print(f"[DEBUG] Uploaded vector store file to GCS: {blob_name}")

    # Upload original PDF to GCS
    upload_to_gcs(file_bytes, pdf_blob_name)
    print(f"[DEBUG] Uploaded PDF to GCS: {pdf_blob_name}")
    print(f"[INFO] Vector store and PDF for {pdf_name} uploaded successfully to GCS.")






def ensure_vectorstores_for_all_pdfs(pdf_files):
    for pdf in pdf_files:
        create_vectorstore_for_pdf(pdf)

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
    gcs_client = storage.Client()

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



@app.get("/search")
async def search_pdfs(query: str = ""):
    results = []
    top_chunks = []

    # 1️⃣ List PDFs and ensure vector stores exist
    pdf_files = list_pdfs(DEMO_FOLDER_ID)
    ensure_vectorstores_for_all_pdfs(pdf_files)

    # 2️⃣ Expand the query for better retrieval
    rewritten_query_prompt = (
        f"Rephrase the following question to make it more specific for finding relevant sections in educational PDFs, "
        f"but keep all the original key words and phrases intact: {query}"
    )
    response = openai_client.chat.completions.create(
        model=REWRITER_MODEL,
        messages=[{"role": "user", "content": rewritten_query_prompt}],
        temperature=0.2
    )
    rewritten_query = response.choices[0].message.content
    print(f"Original query: {query}")
    print(f"Rewritten query for retrieval: {rewritten_query}")

    # 3️⃣ Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.environ.get("OPENAI_API_KEY_S")
    )

    # 4️⃣ Retrieve relevant chunks from each PDF
    for pdf in pdf_files:
        pdf_base_name = pdf["name"].rsplit(".", 1)[0]
        gcs_prefix = f"{pdf_base_name}/vectorstore/"

        # Load vector store
        vectorstore: FAISS = load_vectorstore_from_gcs(gcs_prefix, embeddings)

        # ⚡ Normalize embeddings in FAISS index (if needed)
        if hasattr(vectorstore, "index") and hasattr(vectorstore.index, "normalize_L2"):
            vectorstore.index.normalize_L2()

        # Similarity search with scores (distance-based)
        docs_with_scores = vectorstore.similarity_search_with_score(rewritten_query, k=TOP_K)

        # Collect all chunks with their distance
        for doc, distance_score in docs_with_scores:
            doc.metadata.update({
                "pdf_name": pdf["name"],
                "pdf_base_name": pdf_base_name
            })
            top_chunks.append((doc, distance_score))

    # 5️⃣ Early return if no chunks found
    if not top_chunks:
        results.append({"name": "No results found", "snippet": "", "link": ""})
        return JSONResponse(results)

    # 6️⃣ Sort by ascending distance (least distance = most relevant)
    top_chunks = sorted(top_chunks, key=lambda x: x[1])[:5]  # Take the most relevant

    # 7️⃣ Prepare context for GPT answer
    context_texts = [
        f"PDF: {doc.metadata['pdf_name']}, Page: {doc.metadata['page_number']}\n{doc.page_content}"
        for doc, _ in top_chunks
    ]
    answer_prompt = f"""
You are an assistant. Answer the user question using only the following PDF chunks.
For each fact, indicate the PDF name and page number it came from.

Question: {query}
Chunks:
{chr(10).join(context_texts)}
"""

    answer_response = openai_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0.2
    )
    answer_text = answer_response.choices[0].message.content

    # 8️⃣ Collect PDF links used
    used_pdfs = list({doc.metadata.get("pdf_link") for doc, _ in top_chunks if doc.metadata.get("pdf_link")})

    results.append({
        "name": "Answer",
        "snippet": answer_text,
        "link": ", ".join(used_pdfs)
    })

    return JSONResponse(results)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo_chatbot_backend_2:app", host="0.0.0.0", port=8000, reload=True)
