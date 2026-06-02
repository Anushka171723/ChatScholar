# ChatScholar

ChatScholar is a Flask-based PDF RAG assistant that lets users upload PDFs, ask questions about their documents, and receive AI-powered answers using Groq and FAISS. It also includes an essay grader that provides scores, strengths, weaknesses, and improvement suggestions.

## Features

- Upload one or more PDF files
- Store PDFs in `data/`
- Save and reload FAISS indexes from `faiss_index/`
- Ask document-based questions with source citations
- Keep chat history after browser refresh
- Show a loading state while answers generate
- Grade essays with structured feedback

## Tech Stack

- Flask
- Groq
- FAISS
- PyPDF
- LangChain text splitters

## Project Structure

```text
pdf-rag-assistant/
├── app.py
├── rag.py
├── essay_grader.py
├── utils.py
├── requirements.txt
├── .env
├── README.md
├── data/
├── faiss_index/
├── templates/
└── static/
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create or update `.env`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

`GROQ_MODEL` is optional. The app uses `llama-3.3-70b-versatile` by default.

## Run

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Pages

- `/` - Home
- `/chat` - PDF chat assistant
- `/essay` - Essay grader
- `/health` - App health check

## Notes

Uploaded PDFs are stored in `data/`. The FAISS index is stored in `faiss_index/` and loads automatically when the app starts.
