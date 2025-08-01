# app/main.py
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.backend import extract_text_from_pdf, build_faiss_index, generate_answer
import os
from markdown import markdown as md
import re
import faiss
import numpy as np
app = FastAPI()

def protect_math(text):
    return re.sub(r'(\${2})(.+?)\1', r'@@\2@@', text, flags=re.DOTALL)

def unprotect_math(html):
    return html.replace('@@', '$$')

def auto_wrap_math_blocks(text):
    def wrap_if_needed(line):
        if re.search(r'\\text|\d+\s*[\+\-\*/=]\s*\d+', line) and not re.match(r'\s*\$\$', line):
            return f"$$\n{line.strip()}\n$$"
        return line
    return "\n".join(wrap_if_needed(l) for l in text.splitlines())


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------- Load Static Banking Documents ------------
# ✅ Load saved FAISS index
BASE_DIR = os.path.dirname(__file__)
index_path = os.path.join(BASE_DIR, "vector_base/faiss_index.bin")
faiss_index = faiss.read_index(index_path)

BASE_DIR = os.path.dirname(__file__)
faiss_index = faiss.read_index(os.path.join(BASE_DIR, "vector_base", "faiss_index.bin"))
document_chunks = np.load(os.path.join(BASE_DIR, "vector_base", "chunks.npy"), allow_pickle=True).tolist()



chat_sessions = {}

# -------------- Chat Endpoint ----------------
@app.post("/query")
async def query_endpoint(
    request: Request,
    query: str = Form(...),
    user_data_text: str = Form(""),  # ✅ Accept dynamic user data
):
    client_ip = request.client.host
    chat_history = chat_sessions.get(client_ip, [])

    # This now only uses dynamic user data
    combined_user_data = user_data_text.strip()


    answer = generate_answer(
        query=query,
        index=faiss_index,
        chunks=document_chunks,
        chat_history=chat_history,
        user_data_text=combined_user_data  # ✅ Send to LLM
    )

    chat_history.append({"role": "user", "message": query})
    chat_history.append({"role": "bot", "message": answer})
    chat_sessions[client_ip] = chat_history
    # answer = auto_wrap_math_blocks(answer)
    # markdown_safe = protect_math(answer)
#     html_response = md(
#     answer,
#     extensions=[
#         'markdown.extensions.extra',
#         'markdown.extensions.admonition',
#         'markdown.extensions.codehilite',
#         'markdown.extensions.fenced_code',
#         'markdown.extensions.tables',
#         'markdown.extensions.toc'
#     ]
# )
    # html_response = unprotect_math(html_response)
    return JSONResponse(content={"response": answer})
    
