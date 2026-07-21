# DataLens AI 🔍 — AI-Powered Dataset Assistant

**DataLens AI** adalah aplikasi *AI-powered data assistant* berbasis web yang dirancang untuk membantu pengguna memahami dan mengeksplorasi dataset CSV melalui percakapan bahasa alami (*natural language conversation*). 

Dengan menggunakan pendekatan **Retrieval-Augmented Generation (RAG)**, aplikasi ini secara otomatis membuat profil dataset, menghasilkan *knowledge base* dari metadata data tersebut, lalu mengindeksnya ke dalam *in-memory vector store* (ChromaDB) agar Anda dapat bertanya secara langsung tentang struktur, statistik, dan kualitas data Anda secara instan dan aman.

> **Tagline:** *Understand Your Data. Ask Anything.*

---

## 🌟 Fitur Utama (Core Features)

1. **Dataset Upload & Processing**: Mengunggah file `.csv` secara langsung untuk dianalisis oleh sistem secara lokal menggunakan [Pandas](https://pandas.pydata.org/).
2. **Automatic Data Profiling**: Otomatis menghitung statistik tingkat tinggi seperti total baris/kolom, baris duplikat, *missing cells*, penggunaan memori, serta *Data Quality Health Score*.
3. **Deep Column Inspector**: Melakukan analisis statistik rinci untuk setiap kolom (termasuk *min*, *max*, *mean*, *median*, dan *std* untuk kolom numerik, serta *top frequent values* untuk kolom kategorikal).
4. **RAG Knowledge Base Generation**: Mengonversi profil metadata dataset menjadi dokumen-dokumen penjelasan terstruktur yang ramah bagi LLM (*Large Language Model*).
5. **Dual LLM Provider Support (Gemini & OpenAI)**:
   - **Google Gemini API** (Direkomendasikan/Gratis): Menggunakan `models/gemini-embedding-001` untuk indexing dan `gemini-flash-latest` (Gemini 1.5/2.x) untuk generasi jawaban RAG.
   - **OpenAI API** (Berbayar): Menggunakan `OpenAIEmbeddings` dan `gpt-3.5-turbo` / `gpt-4o`.
6. **Conversational QA with Citations**: Mengajukan pertanyaan tentang dataset dan mendapatkan jawaban grounded yang dilengkapi dengan sumber referensi (*source citations*) konteks dokumen RAG.

---

## 🏗️ Alur Arsitektur (Technical Workflow)

```text
Upload CSV Dataset
       ↓
Automatic Data Profiling (Pandas)
       ↓
Metadata & Health Report Generation
       ↓
Document Creation & Chunking (LangChain)
       ↓
Vector DB Indexing (ChromaDB & Gemini/OpenAI Embeddings)
       ↓
Ask Question (User Input)
       ↓
Context Retrieval (Similarity Search / Vector Store)
       ↓
LLM Answer Synthesis (Gemini/OpenAI Chat LLM)
       ↓
Response with Source Citations
```

---

## 🚀 Cara Menjalankan Secara Lokal (How to Run Locally)

Ikuti langkah-langkah di bawah ini untuk menyiapkan dan menjalankan proyek DataLens AI di komputer lokal Anda.

### 📋 Prasyarat (Prerequisites)
Pastikan Anda sudah menginstal:
* **Python 3.9** atau versi yang lebih baru.
* **API Key** (Dapatkan **Gemini API Key** secara gratis di [Google AI Studio](https://aistudio.google.com/) atau **OpenAI API Key** di [OpenAI Platform](https://platform.openai.com/)).

### 🛠️ Langkah-langkah Setup (Step-by-Step Setup)

1. **Masuk ke Direktori Proyek**:
   Buka terminal/command prompt Anda dan masuk ke folder proyek:
   ```bash
   cd C:/Users/Defanda/Documents/Programming/datalens-ai
   ```

2. **Buat Virtual Environment (Opsional tetapi Direkomendasikan)**:
   Buat lingkungan virtual Python baru agar dependensi tidak bentrok dengan library global:
   ```bash
   python -m venv venv
   ```
   Aktifkan virtual environment tersebut:
   * **Windows (Command Prompt):**
     ```cmd
     venv\Scripts\activate.bat
     ```
   * **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\activate.ps1
     ```
   * **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

3. **Instal Dependensi**:
   Instal semua library yang dibutuhkan sesuai file [requirements.txt]:
   ```bash
   pip install -r requirements.txt
   ```

4. **Konfigurasi Environment Variables**:
   Salin file template [.env.example] menjadi `.env`:
   ```bash
   cp .env.example .env
   ```
   Buka file `.env` yang baru dibuat dan isi API Key yang ingin Anda gunakan:
   ```env
   OPENAI_API_KEY=isi_dengan_api_key_openai_anda
   GOOGLE_API_KEY=isi_dengan_api_key_gemini_anda
   ```

5. **Jalankan Aplikasi Streamlit**:
   Jalankan web server lokal menggunakan perintah streamlit:
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan otomatis terbuka di browser Anda pada alamat default: `http://localhost:8501`.

---

## 📂 Struktur Folder Proyek (Project Layout)

* **[app.py]**: File entry point utama untuk UI Streamlit dan alur logika aplikasi.
* **[src/]**: Folder berisi kode logika backend aplikasi.
  - **[src/profiler.py]**: Logika data profiling otomatis (dataset summary, column profiles).
  - **[src/knowledge_base.py]**: Pembuatan dokumen markdown berdasarkan metadata profil data.
  - **[src/rag_engine.py]**: Konfigurasi RAG pipeline (ChromaDB, Gemini/OpenAI embedding, retrieve, and QA).
  - **[src/ui.py]**: Kustomisasi tampilan CSS, styling kartu metrik, dan sitasi.
* **[sample_data/]**: Folder penyimpanan dataset contoh (seperti Global Superstore Sample) untuk demo cepat.
* **[requirements.txt]**: Daftar pustaka/pustaka pihak ketiga yang wajib diinstal.
