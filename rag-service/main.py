from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
from uuid import uuid4
import os
import time
import torch
import uvicorn
import asyncio
import pdf2image
import pytesseract

# Prompt + post-processing helpers
from utils.postprocess import extract_final_answer, extract_final_summary, extract_comparison
from utils.prompt_templates import build_ask_prompt, build_summarize_prompt, build_compare_prompt


load_dotenv()

# ===============================
# SESSION + CLEANUP
# ===============================
sessions = {}
SESSION_TIMEOUT = 3600


def cleanup_expired_sessions():
    now = time.time()
    expired = [
        sid for sid, data in sessions.items()
        if now - data["last_accessed"] > SESSION_TIMEOUT
    ]
    for sid in expired:
        del sessions[sid]


async def background_cleanup():
    while True:
        await asyncio.sleep(300)
        cleanup_expired_sessions()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(background_cleanup())
    yield
    task.cancel()


# ===============================
# FASTAPI APP
# ===============================
app = FastAPI(
    title="PDF QA Bot API",
    version="2.3.0",
    lifespan=lifespan
)

# ===============================
# CORS
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# RATE LIMIT
# ===============================
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ===============================
# EMBEDDING MODEL
# ===============================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ===============================
# GENERATION MODEL
# ===============================
HF_GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "google/flan-t5-small")

config = AutoConfig.from_pretrained(HF_GENERATION_MODEL)
is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
tokenizer = AutoTokenizer.from_pretrained(HF_GENERATION_MODEL)

if is_encoder_decoder:
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_GENERATION_MODEL)
else:
    model = AutoModelForCausalLM.from_pretrained(HF_GENERATION_MODEL)

if torch.cuda.is_available():
    model = model.to("cuda")

model.eval()


def generate_response(prompt: str, max_new_tokens: int = 200):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    if is_encoder_decoder:
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


# ===============================
# REQUEST MODELS
# ===============================
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_ids: list = []


class SummarizeRequest(BaseModel):
    session_ids: list = []


class CompareRequest(BaseModel):
    session_ids: list = []


# ===============================
# HEALTH
# ===============================
@app.get("/healthz")
def health_check():
    return {"status": "healthy"}


# ===============================
# UPLOAD + OCR + SECURITY
# ===============================
@app.post("/upload")
@limiter.limit("10/15 minutes")
async def upload_file(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF supported"}

    session_id = str(uuid4())

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, f"{uuid4().hex}.pdf")

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        docs = PyPDFLoader(file_path).load()

        # OCR fallback
        final_docs = []
        images = None

        for i, doc in enumerate(docs):
            if len(doc.page_content.strip()) < 50:
                if images is None:
                    images = pdf2image.convert_from_path(file_path)

                ocr_text = pytesseract.image_to_string(images[i])
                final_docs.append(
                    Document(
                        page_content=ocr_text,
                        metadata={"page": i}
                    )
                )
            else:
                final_docs.append(doc)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )

        chunks = splitter.split_documents(final_docs)

        if not chunks:
            return {"error": "No extractable text"}

        vectorstore = FAISS.from_documents(chunks, embedding_model)

        sessions[session_id] = {
            "vectorstores": [vectorstore],
            "filename": file.filename,
            "last_accessed": time.time()
        }

        return {
            "message": "Upload successful",
            "session_id": session_id
        }

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ===============================
# ASK
# ===============================
@app.post("/ask")
@limiter.limit("60/15 minutes")
def ask(request: Request, data: AskRequest):
    cleanup_expired_sessions()

    if not data.session_ids:
        return {"answer": "No session selected."}

    docs_meta = []

    for sid in data.session_ids:
        session = sessions.get(sid)
        if not session:
            continue

        session["last_accessed"] = time.time()
        vs = session["vectorstores"][0]

        retrieved = vs.similarity_search(data.question, k=4)

        for d in retrieved:
            docs_meta.append({
                "doc": d,
                "filename": session.get("filename", "unknown")
            })

    if not docs_meta:
        return {"answer": "No relevant info"}

    context = "\n\n".join(
        f"[Page {d['doc'].metadata.get('page',0)+1}] {d['doc'].page_content}"
        for d in docs_meta
    )

    prompt = build_ask_prompt(context=context, question=data.question)
    raw = generate_response(prompt)
    answer = extract_final_answer(raw)

    return {"answer": answer}


# ===============================
# SUMMARIZE
# ===============================
@app.post("/summarize")
def summarize(data: SummarizeRequest):
    contexts = []

    for sid in data.session_ids:
        session = sessions.get(sid)
        if session:
            vs = session["vectorstores"][0]
            chunks = vs.similarity_search("summary", k=6)
            contexts.extend(c.page_content for c in chunks)

    prompt = build_summarize_prompt(context="\n\n".join(contexts))
    raw = generate_response(prompt, 300)

    return {"summary": extract_final_summary(raw)}


# ===============================
# COMPARE (FIXED BUG)
# ===============================
@app.post("/compare")
def compare(data: CompareRequest):
    if len(data.session_ids) < 2:
        return {"comparison": "Select at least 2 docs"}

    per_doc_contexts = []

    for sid in data.session_ids:
        session = sessions.get(sid)
        if not session:
            continue

        vs = session["vectorstores"][0]
        chunks = vs.similarity_search("main topic", k=4)
        text = "\n".join(c.page_content for c in chunks)
        per_doc_contexts.append(text)

    prompt = build_compare_prompt(per_doc_contexts=per_doc_contexts)
    raw = generate_response(prompt, 400)

    return {"comparison": extract_comparison(raw)}


# ===============================
# START
# ===============================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)