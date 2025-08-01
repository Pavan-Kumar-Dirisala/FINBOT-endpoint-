import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import traceback
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
# Global variable to cache last successful API index during runtime
current_api_index = 0

# ✅ Load multiple OpenRouter API keys from environment
openrouter_keys = os.getenv("OPENROUTER_API_KEYS", "").split(",")

# ✅ Preferred model
# MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b:free"
# MODEL_NAME = "qwen/qwen3-4b:free"
MODEL_NAME = "qwen/qwen-2.5-72b-instruct:free"
# MODEL_NAME = "google/gemma-3-12b-it:free"

# ✅ Load sentence embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ✅ Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return " ".join([page.get_text() for page in doc])


# ✅ Step 2: Build FAISS index from text chunks
def build_faiss_index(text, chunk_size=500):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks


# ✅ Step 3: Generate answer with OpenRouter fallback
def generate_answer(query, index, chunks, chat_history=None, user_data_text="", top_k=10):
    global current_api_index  # Cache index across calls

    # Vectorize query and retrieve top-k matching chunks
    query_vec = embedding_model.encode([query])
    _, indices = index.search(np.array(query_vec), top_k)
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]

    # Warn if low-quality retrieval
    if not any(relevant_chunks):
        return "⚠️ I'm unable to find any relevant context to answer your question accurately. Please try rephrasing or ask a different question."

    context = "\n".join(relevant_chunks)

    # Chat history formatting
    history_block = ""
    if chat_history:
        for turn in chat_history:
            who = "User" if turn["role"] == "user" else "FinBot"
            history_block += f"{who}: {turn['message']}\n"

    # Optional user-specific data
    user_data_block = f"\n<user_data>\n{user_data_text.strip()}\n</user_data>" if user_data_text else ""
    additional_prompt = "Use the user's name from <user_data> when appropriate. Always prioritize <context> and <user_data> while answering." if user_data_text else ""

    # Final prompt to model
    prompt = f"""
<system>
You are FinBot, a respectful, helpful, and context-aware financial assistant. 
Your answers must be grounded in the provided <context> and <user_data>. 
Do not hallucinate or fabricate. If relevant information is missing, clearly state that.
But ack as a best friend and give suggestions
</system>

<additional_prompt>
{additional_prompt}
</additional_prompt>

<chat_history>
{history_block}
</chat_history>

<context>
{context}
</context>
{user_data_block}

<question>
{query}
</question>

<instructions>
- Answer precisely using <context> and <user_data>.
- provide suggestions and actionalble insights also.
- You have to answer only related to the bank documents you are given in the <context> don,t refer any other banks.
- If <context> doesn’t contain the answer, say it clearly and suggest possible next steps.
- if the queries are genral grewints or praisings like that respond with a positive tone.
- Maintain a professional and empathetic tone.
- only if the user asks for a specific financial product or service, provide a clear and concise description of it.
- Never make up information. Never assume. Be transparent.
- if the query is to compare then give tabular form and explain that as paragraph.
- at the end ask for few followeing up questions that you think are relevant to the query.
- Format using Markdown:
  - Use `##`, `###` for headings.
  - Use `**bold**` for highlights.
  - Use `-` for bullet points.
  - 
</instructions>
"""

    # Try all available API keys
    indices_to_try = [current_api_index] + [i for i in range(len(openrouter_keys)) if i != current_api_index]

    for idx in indices_to_try:
        api_key = openrouter_keys[idx].strip()
        try:
            print(f"🔑 Trying API Key #{idx + 1}")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                extra_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "FinBot"
                }
            )
            print(f"✅ Successfully used API Key #{idx + 1}")
            current_api_index = idx  # Update cache
            # print("🔍 Raw completion:", completion)
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Failed with API Key #{idx + 1}: {e}")
            traceback.print_exc()

    return "❗ All OpenRouter API keys failed. Please try again later."
