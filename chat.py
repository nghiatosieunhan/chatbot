import os
from src.chatbot_agentic import OptimizedAgenticRAG
from dotenv import load_dotenv

def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "vectorstore", "chromadb")
    
    try:
        bot = OptimizedAgenticRAG(db_path=DB_PATH, api_key=api_key)
        print("\n--- NHẬP 'EXIT' ĐỂ THOÁT ---")
        
        while True:
            query = input("\n👤 Bạn: ")
            if query.lower() in ['exit', 'quit']: break
            
            result = bot.ask(query)
            print(f"\n🤖 Bot: {result['answer']}")
            
    except Exception as e:
        print(f"❌ Lỗi khởi động: {e}")

if __name__ == "__main__":
    main()