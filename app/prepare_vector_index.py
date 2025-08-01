import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
DOCUMENTS_FOLDER = "documents"
OUTPUT_INDEX_FILE = "faiss_index.bin"
OUTPUT_CHUNKS_FILE = "chunks.npy"
CHUNK_SIZE = 500

# ---------- Embedding model ----------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Step 1: Extract text from all PDFs ----------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return " ".join([page.get_text() for page in doc])

all_text = ""
for filename in os.listdir(DOCUMENTS_FOLDER):
    if filename.endswith(".pdf") and filename != "user_data.pdf":
        full_path = os.path.join(DOCUMENTS_FOLDER, filename)
        print(f"📄 Extracting: {filename}")
        all_text += extract_text_from_pdf(full_path) + "\n"

# ---------- Step 2: Chunk text ----------
chunks = [all_text[i:i + CHUNK_SIZE] for i in range(0, len(all_text), CHUNK_SIZE)]

# ---------- Step 3: Embed and build FAISS index ----------
print("🔍 Generating embeddings...")
embeddings = embedding_model.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# ---------- Step 4: Save index and chunks ----------
print("💾 Saving FAISS index and chunks...")
faiss.write_index(index, OUTPUT_INDEX_FILE)
np.save(OUTPUT_CHUNKS_FILE, chunks, allow_pickle=True)

print("✅ Done. You can now delete the PDF files from deployment.")
