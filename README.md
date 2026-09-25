# Document Atlas: PDF RAG Question Answering Bot

Document Atlas is a Django-based Retrieval-Augmented Generation (RAG) application for asking questions about local PDF, TXT, and DOCX documents. It extracts document text, retrieves the most relevant passages with FAISS, and uses Groq to produce a concise answer grounded only in the retrieved context.

The included knowledge base currently contains `Computer-Network-Notes-complete.pdf` and `Notes.pdf`. The application returns the current question and its generated answer through the web interface and also provides a terminal command for direct queries.

## Tech stack

| Technology | Version requirement in this project | Purpose |
| --- | --- | --- |
| Python | Python 3 (project environment uses Python 3.12) | Application runtime |
| Django | `>=5.0,<6.0` | Web application and API routing |
| python-dotenv | `>=1.0.1` | Loads variables from `.env` |
| PyMuPDF | `>=1.24.9` | PDF text extraction |
| python-docx | `>=1.1.2` | DOCX text extraction |
| sentence-transformers | `>=3.2.1` | Text and query embeddings |
| FAISS CPU | `>=1.8.0` | Local vector similarity search |
| Groq Python SDK | `>=0.9.0` | Grounded answer generation |
| pytest | `>=8.3.2` | Test runner |
| pytest-django | `>=4.8.0` | Django test integration |

## Architecture

```text
PDF, TXT, and DOCX files in data/
       |
       v
Document ingestion (PyMuPDF for PDF, python-docx for DOCX; filename and page metadata retained)
       |
       v
Overlapping text chunking
       |
       v
Embeddings (sentence-transformers/all-MiniLM-L6-v2)
       |
       v
Persistent FAISS IndexFlatIP vector store
       |
       v
TOP_K retrieval for the current question
       |
       v
Groq generation using only retrieved context
       |
       v
Grounded natural-language answer
```

The index is persisted in `vectorstore/faiss_index.index` with accompanying metadata in `vectorstore/metadata.json`. Before use, the application compares the current PDF file signature with `vectorstore/document_state.json`; unchanged documents reuse the existing index, while changed documents are re-extracted and re-indexed.

## Chunking strategy

PDFs are extracted page by page; TXT and DOCX documents are each represented as one source page. Whitespace is normalized before every source page is split with a fixed character-window strategy:

- Chunk size: **500 characters**
- Chunk overlap: **100 characters**
- Step size: **400 characters**

The overlap preserves context across adjacent chunk boundaries while keeping chunks small enough for targeted vector retrieval. Each chunk retains its original filename, page number, and a chunk ID in the form `filename::p<page>::c<chunk>`.

## Embeddings and vector database

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector database: local FAISS `IndexFlatIP`

The embedding model converts document chunks and the current question into vectors of the same dimension. FAISS stores those vectors locally and performs inner-product similarity search. The application retrieves the configurable `TOP_K` best matches (default `4`) without applying a fixed similarity-score rejection threshold. The embedding model is loaded from the local cache during requests.

## Setup

### 1. Clone the repository

Replace `<repository-url>` with the repository URL provided for this project.

```powershell
git clone <repository-url>
cd document_qa_bot
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file and edit the new `.env` file:

```powershell
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` to your own Groq API key. Do not commit `.env` and never place a real key in this README or in `.env.example`.

### 5. Add documents

Place supported `.pdf`, `.txt`, or `.docx` files in the `data/` directory. The repository currently includes computer-network and notes PDFs as example documents.

### 6. Index documents

There is no separate indexing command. Indexing runs automatically before a question is answered:

```powershell
python manage.py ask_rag "What is a computer network?"
```

On the first run, or when PDFs in `data/` change, the application extracts text, chunks pages, creates embeddings, and saves the FAISS index. On later runs with unchanged PDFs, it reuses the saved index.

### 7. Run the application

```powershell
python manage.py runserver
```

Open `http://127.0.0.1:8000/` in a browser and ask a question. The API endpoint is available at `/ask/` and accepts the current question in a POST body or form field.

To ask from the terminal instead:

```powershell
python manage.py ask_rag "What is a computer network?"
```

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `DJANGO_SECRET_KEY` | No | Django secret key. The application has a development fallback, but set a unique value for deployment. |
| `DEBUG` | No | Enables Django debug mode when set to `True`, `1`, `yes`, or `on`. |
| `GROQ_API_KEY` | Yes for generated answers | Groq API key used for answer generation. Keep it private. |
| `GROQ_MODEL` | No | Groq generation model. The example configuration uses `openai/gpt-oss-20b`. |
| `TOP_K` | No | Number of FAISS matches to retrieve; defaults to `4`. |
| `RAG_DEBUG` | No | Controls detailed RAG pipeline logging. The backend defaults this logging to enabled when the variable is absent. |

Use `.env.example` as the template. It intentionally contains only a placeholder key:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

## Sample questions

The expected topic of each answer depends only on information in the indexed PDFs.

| Sample question | Expected answer topic |
| --- | --- |
| `What is a computer network?` | Definition of a computer network and its connected components. |
| `What are the advantages of a computer network?` | Benefits such as file/resource sharing described in the network notes. |
| `What is a client-server network?` | Client-server authentication, centralization, and administration. |
| `What is TCP?` | Reliable, connection-oriented TCP data transmission. |
| `What is the difference between a datagram network and a virtual circuit network?` | How packet forwarding differs between the two network types. |
| `What is quantum computing?` | Expected fallback response when the indexed documents do not provide enough supporting context. |

## Known limitations

- DOCX and TXT documents are represented as a single source page because the current extraction layer does not preserve their visual page boundaries.
- Chunking is fixed to character windows; it does not use semantic or sentence-aware splitting.
- The embedding model must already be available in the local model cache because requests use `local_files_only=True`.
- The quality and completeness of answers depend on text extraction and the content of the indexed PDFs.
- FAISS retrieval is local and uses a flat inner-product index; it is appropriate for the current small local corpus but has no remote database, authentication, or multi-user isolation.
- Groq availability, a valid API key, and an available configured model are required for generated answers. If generation fails or the retrieved context is insufficient, the application returns its document-only unknown-answer fallback.
- Document updates are detected from PDF file path, size, and modification time; re-indexing happens when that signature changes.

## Requirement coverage checklist

- [x] Project title and description
- [x] Technology stack with project version requirements
- [x] Ingestion-to-generation architecture diagram
- [x] Chunking method, size, overlap, and rationale
- [x] Embedding model and vector database details
- [x] Clone, install, environment, document, indexing, and run instructions
- [x] Environment-variable and API-key guidance
- [x] Six sample questions with expected topics
- [x] Known limitations
