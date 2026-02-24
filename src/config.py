import torch
import os

# Tự động chọn thiết bị: ưu tiên GPU (cuda) nếu có
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CHẾ ĐỘ CHẠY: Thay đổi giữa 'LITE' và 'PRO'
# LITE: Dùng cho máy Local (yếu) - Tắt BM25, dùng Reranker nhẹ.
# PRO: Dùng cho Colab hoặc máy mạnh - Bật Hybrid Search (BM25 + Vector) và Reranker nặng.
RUN_MODE = "LITE" 

if RUN_MODE == "PRO":
    EMBEDDING_MODEL = "models/text-embedding-004"
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # 2.2GB
    USE_BM25 = True
    TOP_K_RETRIVE = 50
else:
    EMBEDDING_MODEL = "models/gemini-embedding-001"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 80MB
    USE_BM25 = False # Tránh lỗi sập RAM ở Step 4
    TOP_K_RETRIVE = 30