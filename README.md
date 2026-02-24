# 🤖 Adaptive Hybrid RAG Chatbot: Cloud-First Pipeline

[![Model](https://img.shields.io/badge/Model-Gemini%202.0%20Flash-orange.svg)](https://aistudio.google.com/)
[![Tech Stack](https://img.shields.io/badge/Stack-Python%20%7C%20ChromaDB%20%7C%20LangChain-blue.svg)](#tech-details)
[![Deployment](https://img.shields.io/badge/Deployment-Colab%20%7C%20Kaggle-success.svg)](#cloud-deployment)
[![University](https://img.shields.io/badge/USTH-Information%20Security-red.svg)](https://usth.edu.vn/)

> **🎯 Giải pháp RAG Tối ưu Cloud**
> Xây dựng hệ thống phân tích báo cáo tự động: PDF thô → Text sạch → Vector DB → Chatbot. Được thiết kế và tối ưu hóa để bung sức mạnh tối đa trên Google Colab/Kaggle.

---

## ⚙️ **1. Chọn chế độ vận hành (Mode Selection)**

Dự án có 2 chế độ cấu hình tại `src/config.py`. Việc chọn đúng Mode giúp tránh lỗi tràn bộ nhớ (Out of Memory).

* **🔥 Chế độ PRO (Khuyên dùng)**: Dành cho Colab, Kaggle hoặc máy tính có GPU.
    * Kích hoạt **Hybrid Search** (Vector + BM25).
    * Sử dụng Reranker hạng nặng `bge-reranker-v2-m3` (2.2GB) để đạt độ chính xác tối đa.
* **🧊 Chế độ LITE**: Dành cho máy cá nhân (RAM < 16GB). 
    * Chỉ dùng Vector Search và Reranker mini (80MB) để test logic code mà không gây crash máy.

---

## ☁️ **2. Triển khai trên Cloud (Google Colab / Kaggle)**

Đây là môi trường **được khuyến nghị** để chạy dự án nhằm tận dụng GPU miễn phí, giúp mô hình nhúng và Reranker hoạt động với tốc độ cao nhất.

### **Bước 1: Thiết lập phần cứng**
* **Google Colab**: `Runtime` -> `Change runtime type` -> Chọn **T4 GPU**.



### **Bước 2: Clone dự án & Cài đặt**
Mở một Notebook mới (Cell đầu tiên) và chạy:
```bash
!git clone [https://github.com/nghiatosieunhan/chatbot.git](https://github.com/nghiatosieunhan/chatbot.git)
%cd chatbot
!pip install -r requirements.txt
```

### **Bước 3: Cấu hình API Key (Bảo mật)**
Sử dụng tính năng bảo mật của nền tảng để lưu Key, tuyệt đối không gán cứng (hard-code) vào file:
* **Colab**: Lưu vào mục **Secrets** (biểu tượng chìa khóa bên trái) với tên `GOOGLE_API_KEY` và `LANDING_AI_KEY`.

### **Bước 4: Chạy Pipeline**
```bash
# 1. Trích xuất Text từ file PDF thô (đặt trong data/raw/)
!python pdf_to_txt.py

# 2. Xây dựng Database Vector (Chế độ PRO)
!python build.py

# 3. Mở giao diện Chat (trên Terminal của Notebook)
!python chat.py
```

---

## 💻 **3. Triển khai Local (Tùy chọn)**

Nếu bạn muốn chạy trực tiếp trên máy tính cá nhân để test giao diện hoặc debug:

1. Đảm bảo cấu hình trong `src/config.py` đang là **LITE**.
2. Thiết lập môi trường ảo và cài thư viện:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   echo "GOOGLE_API_KEY=your_key_here" > .env
   ```
3. Chạy theo thứ tự:
   ```bash
   python pdf_to_txt.py
   python build.py
   python app_gui.py  # Mở giao diện đồ họa
   ```

---

## 🔄 **4. Workflow Hệ thống**



Hệ thống hoạt động theo đường ống khép kín:
1.  **Ingestion**: Nhận file PDF từ `data/raw/`.
2.  **Conversion**: `pdf_to_txt.py` làm sạch và chuyển đổi sang `.txt` lưu tại `data/processed/`.
3.  **Indexing**: `build.py` cắt nhỏ văn bản (chunking) và tạo Vector Index bằng ChromaDB.
4.  **Hybrid Retrieval**: Tìm kiếm đa tầng kết hợp ngữ nghĩa và từ khóa.
5.  **Reranking**: Sắp xếp lại mức độ ưu tiên của các đoạn văn bản.
6.  **Generation**: Gemini tổng hợp và trả lời người dùng.

---

## 📁 **5. Cấu trúc thư mục dự án**

```text
baocao_chatbot/
├── 📂 data/
│   ├── 📂 raw/               # 📥 File PDF gốc (chứa .gitkeep)
│   └── 📂 processed/         # 📄 File Text sạch sau convert (chứa .gitkeep)
├── 📂 src/
│   ├── config.py             # ⚙️ Trung tâm điều khiển LITE/PRO
│   └── chatbot_agentic.py    # 🧠 Logic xử lý RAG
├── 📂 vectorstore/           # 💾 CSDL ChromaDB (Đã chặn bởi gitignore)
├── app_gui.py                # 🖼️ Giao diện đồ họa (Chạy Local)
├── build.py                  # 🔨 Script nạp dữ liệu vào Database
├── chat.py                   # 💬 Giao diện chat Terminal
├── pdf_to_txt.py             # 🛠️ Script chuyển đổi PDF sang Text
├── requirements.txt          # 📋 Danh sách thư viện cần thiết
└── .gitignore                # 🛡️ Bảo vệ API Key và dữ liệu nặng
```

---

## 🛡️ **6. Ghi chú Bảo mật (Infosec)**
* **Data Privacy**: Các thư mục nhạy cảm (`data/raw/` chứa báo cáo thật và `vectorstore/` chứa dữ liệu đã mã hóa) được cấu hình trong `.gitignore` để tránh đẩy lên GitHub Public.
* **Key Management**: Quản lý khóa API nghiêm ngặt qua file `.env` (Local) và Secrets (Cloud).

---
**⭐ Nếu bạn thấy dự án này hữu ích, hãy tặng một Star nhé!**