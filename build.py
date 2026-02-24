import os
import re
import uuid
import time
import chromadb
import torch
import numpy as np
from chromadb.utils import embedding_functions
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from tqdm import tqdm 
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COLLECTION_NAME = "baocao_chroma"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed") 
DB_PATH = os.path.join(BASE_DIR, "vectorstore", "chromadb")

def clean_content(text, is_markdown=True):
    """Combining cleaning logic from both versions."""
    if is_markdown:
        text = re.sub(r'', '', text, flags=re.DOTALL)
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        text = re.sub(r'(?<=[a-z,])\n+(?=[a-z])', ' ', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
    else:
        text = re.sub(r'Trang \d+/\d+', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'', '', text, flags=re.DOTALL)
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        text = re.sub(r'(?<=[a-z,])\n+(?=[a-z])', ' ', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()

def batch_insert(collection, docs, metas, ids, batch_size=100): 
    total = len(docs)
    print(f"Saving {total} chunks...")

    for i in tqdm(range(0, total, batch_size), desc="Uploading datas to Cloud"):
        end = min(i + batch_size, total)
        batch_docs = docs[i:end]
        batch_metas = metas[i:end]
        batch_ids = ids[i:end]
        
        attempt = 0
        while attempt < 3:
            try:
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                break 
            except Exception as e:
                if "429" in str(e):
                    wait_time = 5 * (attempt + 1)
                    print(f"Limit quota, wait {wait_time}s...")
                    time.sleep(wait_time)
                    attempt += 1
                else:
                    print(f"\nError: {e}")
                    break
def main():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=GOOGLE_API_KEY
    )
    gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=GOOGLE_API_KEY,
        task_type="RETRIEVAL_DOCUMENT",
        model_name="models/gemini-embedding-001"
    )
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=gemini_ef)

    txt_splitter = SemanticChunker(
        embeddings=embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    if not os.path.exists(DATA_DIR):
        print(f"❌ Thư mục không tồn tại: {DATA_DIR}")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.md', '.txt'))]
    print(f"🔍 Find {len(files)} file: {files}")

    all_docs, all_metas, all_ids = [], [], []

    for file_name in files:
        file_path = os.path.join(DATA_DIR, file_name)
        is_md = file_name.endswith('.md')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned = clean_content(content, is_markdown=is_md)
        splitter = txt_splitter
        chunks = txt_splitter.create_documents([cleaned])
        
        for chunk in chunks:
            all_docs.append(chunk.page_content)
            all_metas.append({"source": file_path, "format": "md" if is_md else "txt"})
            all_ids.append(str(uuid.uuid4()))

    if all_docs:
        batch_insert(collection, all_docs, all_metas, all_ids)
        print(f"Uploaded {len(all_docs)} chunks to {DB_PATH}")
    else:
        print("No data to upload.")

if __name__ == "__main__":
    main()