import chromadb
import os

# Đường dẫn đúng tính từ thư mục gốc dự án
DB_PATH = "./vectorstore/chromadb"

def diagnostic_check():
    print(f"--- Checking ChromaDB Integrity ---")
    abs_path = os.path.abspath(DB_PATH)
    print(f"📂 Target Path: {abs_path}")

    if not os.path.exists(abs_path):
        print(f"❌ Error: Folder not found! Current working dir: {os.getcwd()}")
        return

    try:
        # Khởi tạo client
        client = chromadb.PersistentClient(path=abs_path)
        
        # Liệt kê các collections (bảng dữ liệu)
        collections = client.list_collections()
        print(f"📊 Found {len(collections)} collection(s).")
        
        for coll_info in collections:
            coll = client.get_collection(name=coll_info.name)
            count = coll.count()
            print(f"\n🔹 Collection Name: '{coll_info.name}'")
            print(f"✅ Total Items (Chunks): {count}")
            
            if count > 0:
                print("👀 Peeking at the first chunk:")
                peek = coll.peek(1)
                print(f"   - ID: {peek['ids'][0]}")
                print(f"   - Metadata: {peek['metadatas'][0]}")
                print(f"   - Content: {peek['documents'][0][:100]}...")
    
    except Exception as e:
        print(f"❌ Critical Error reading DB: {e}")

if __name__ == "__main__":
    diagnostic_check()