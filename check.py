import os
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb

load_dotenv()

def test_components():
    print("Step 1: Testing Google API Connection...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        # Thử tính toán một vector ngắn để xem có bị treo không
        vector = embeddings.embed_query("hello")
        print("✅ API Connection: OK")
    except Exception as e:
        print(f"❌ API Connection Failed: {e}")
        return

    print("\nStep 2: Testing ChromaDB Access...")
    # SỬA ĐƯỜNG DẪN NÀY CHO ĐÚNG VỚI ẢNH CỦA BẠN
    db_path = os.path.abspath("./vectorstore/chromadb")
    print(f"📂 Accessing: {db_path}")
    
    try:
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        print(f"✅ ChromaDB Access: OK (Found {len(collections)} collections)")
    except Exception as e:
        print(f"❌ ChromaDB Failed: {e}")

    print("\nStep 3: Checking Reranker (If any)...")
    print("If it hangs here, your Reranker model is too heavy or downloading.")
    # Nếu bạn có dùng CrossEncoder, hãy thêm dòng test ở đây

if __name__ == "__main__":
    test_components()