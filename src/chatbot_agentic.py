import os, re, numpy as np, torch, chromadb, gc
from rank_bm25 import BM25Okapi 
from google import genai
from sentence_transformers import CrossEncoder
from src.config import DEVICE, RERANKER_MODEL, EMBEDDING_MODEL, USE_BM25

class GenAIEmbeddingFunction:
    def __init__(self, client):
        self.client = client
    def __call__(self, input):
        texts = [input] if isinstance(input, str) else input
        res = self.client.models.embed_content(
            model=EMBEDDING_MODEL, 
            contents=texts,
            config={'task_type': 'RETRIEVAL_QUERY'}
        )
        return [e.values for e in res.embeddings]
    def embed_query(self, input_text):
        return self.__call__(input_text)[0]

class OptimizedAgenticRAG:
    def __init__(self, db_path, api_key):
            self.client = genai.Client(api_key=api_key)
            self.model_name = "gemini-2.0-flash" 
            self.db_path = db_path
            
            self.reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE)
            self.embedding_fn = GenAIEmbeddingFunction(self.client)
            
            try:
                self.chroma_client = chromadb.PersistentClient(path=self.db_path)
                self.collection = self.chroma_client.get_collection(name="baocao_chroma")
            except Exception as e:
                print(f"❌ Lỗi bước 3: Không tìm thấy collection 'baocao_chroma' trong {self.db_path}")
                print("👉 Hãy chắc chắn bạn đã chạy build.py vào đúng folder này.")
                raise e

            if USE_BM25:
                all_docs = self.collection.get()
                self.documents = all_docs.get('documents', [])
                self.bm25 = BM25Okapi([doc.split() for doc in self.documents])
            else:
                self.bm25 = None
            print("✅ Agentic RAG đã sẵn sàng!")
    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text) 
        return text.split()

    def _deduplicate_contexts(self, contexts):
        seen = set()
        unique = []
        for ctx in contexts:
            ctx_hash = ctx[:100]
            if ctx_hash not in seen:
                seen.add(ctx_hash)
                unique.append(ctx)
        return unique

    def hybrid_search(self, query, top_k=10):
        query_emb = self.embedding_fn.embed_query(query)
        v_docs = self.collection.query(
            query_embeddings=[query_emb], 
            n_results=40)['documents'][0]
        
        if self.bm25:
            b_docs = self.bm25.get_top_n(
                self._tokenize(query), 
                self.documents, n=20)
        else:
            b_docs = []  
        
        candidate_docs = list(set(v_docs + b_docs))
        candidate_docs = self._deduplicate_contexts(candidate_docs)
        
        scores = self.reranker.predict([[query, d] for d in candidate_docs], batch_size=16)
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        
        top_indices = np.argsort(scores)[::-1]
        contexts = [
            candidate_docs[i] 
            for i in top_indices[:top_k] 
            if scores[i] > 0.25]
        return contexts

    def ask(self, question):
        try:
            contexts = self.hybrid_search(question)
            context_text = "\n\n".join(contexts)
            prompt = f"""Bạn là chuyên gia phân tích báo cáo của thành phố Hà Nội.

NHIỆM VỤ: Trả lời câu hỏi dựa HOÀN TOÀN vào thông tin trong CONTEXT bên dưới.

QUY TẮC:
1. Chỉ sử dụng thông tin từ CONTEXT
2. Nếu có bảng số liệu, trích xuất chính xác
3. Trả lời ngắn gọn, súc tích
4. Nếu không tìm thấy thông tin, nói rõ "Không có thông tin trong báo cáo"

CONTEXT:
{context_text}

CÂU HỎI: {question}

TRẢ LỜI:"""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={'temperature': 0.1, 'max_output_tokens': 512}
            )
            return {"answer": response.text.strip(), "retrieved_contexts": contexts}
        except Exception as e:
            return {"answer": f"❌ Lỗi: {str(e)}", "retrieved_contexts": []}
    