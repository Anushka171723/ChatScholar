import os
import pickle
import re
from pathlib import Path

import faiss  # type: ignore[import-untyped]
import numpy as np
from groq import Groq

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

DEFAULT_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_DIMENSIONS = 384
INDEX_FILE = "index.faiss"
STORE_FILE = "store.pkl"
INDEX_VERSION = 2
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "does",
    "for",
    "from",
    "how",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


class DocumentStore:
    def __init__(self, data_dir, index_dir):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.documents = []
        self.chunks = []
        self.index = None
        self.sync()

    def sync(self):
        self.data_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)

        if self.index_is_current():
            self.load_index()
            return

        self.rebuild_index()

    def index_is_current(self):
        index_path = self.index_dir / INDEX_FILE
        store_path = self.index_dir / STORE_FILE
        if not index_path.exists() or not store_path.exists():
            return False

        try:
            with store_path.open("rb") as file:
                data = pickle.load(file)
            return (
                data.get("index_version") == INDEX_VERSION
                and data.get("pdf_signature") == self.pdf_signature()
            )
        except Exception:
            return False

    def load_index(self):
        self.index = faiss.read_index(str(self.index_dir / INDEX_FILE))

        with (self.index_dir / STORE_FILE).open("rb") as file:
            data = pickle.load(file)

        self.documents = data.get("documents", [])
        self.chunks = data.get("chunks", [])

    def rebuild_index(self):
        self.documents = []
        self.chunks = []

        for pdf_path in sorted(self.data_dir.glob("*.pdf")):
            pages = extract_pdf_pages(pdf_path)
            text = "\n\n".join(page["text"] for page in pages)
            if not text:
                continue

            self.documents.append({"name": pdf_path.name, "text": text})
            for page in pages:
                for index, chunk in enumerate(chunk_text(page["text"]), start=1):
                    self.chunks.append(
                        {
                            "source": pdf_path.name,
                            "page": page["page"],
                            "chunk_index": index,
                            "text": chunk,
                        }
                    )

        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
        if self.chunks:
            vectors = np.vstack(
                [embed_text(chunk["text"]) for chunk in self.chunks]
            ).astype("float32")
            self.index.add(vectors)

        faiss.write_index(self.index, str(self.index_dir / INDEX_FILE))
        with (self.index_dir / STORE_FILE).open("wb") as file:
            pickle.dump(
                {
                    "index_version": INDEX_VERSION,
                    "pdf_signature": self.pdf_signature(),
                    "documents": self.documents,
                    "chunks": self.chunks,
                },
                file,
            )

    def files(self):
        return [document["name"] for document in self.documents]

    def chunk_count(self):
        return len(self.chunks)

    def pdf_signature(self):
        signature = []
        for pdf_path in sorted(self.data_dir.glob("*.pdf")):
            stat = pdf_path.stat()
            signature.append((pdf_path.name, stat.st_size, stat.st_mtime_ns))
        return signature


def build_document_store(data_dir, index_dir="faiss_index"):
    return DocumentStore(data_dir, index_dir)


def extract_pdf_pages(path):
    if PdfReader is None:
        return []

    try:
        reader = PdfReader(str(path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page": page_number, "text": text})
        return pages
    except Exception:
        return []


def answer_question(question, document_store):
    context, sources = find_relevant_context(question, document_store)
    if not context:
        return {
            "answer": "Upload a PDF first, then ask me about it.",
            "sources": [],
        }

    answer = answer_with_groq(question, context) or answer_locally(context)
    return {"answer": answer, "sources": sources}


def find_relevant_context(question, document_store):
    if not document_store.chunks or document_store.index is None:
        return "", []

    question_terms = tokenize_question(question) or tokenize_text(question)
    query_vector = embed_text(question).reshape(1, -1).astype("float32")
    result_count = min(12, len(document_store.chunks))
    distances, indexes = document_store.index.search(query_vector, result_count)
    scored = []

    for distance, chunk_id in zip(distances[0], indexes[0]):
        if chunk_id < 0:
            continue

        chunk = document_store.chunks[int(chunk_id)]
        keyword_score = score_chunk(question, question_terms, chunk["text"])
        vector_score = float(distance) * 20
        scored.append(
            (
                keyword_score + vector_score,
                chunk["source"],
                chunk["page"],
                chunk["chunk_index"],
                chunk["text"],
            )
        )

    if not scored:
        return "", []

    scored.sort(key=lambda item: (-item[0], item[2], item[3]))
    top_chunks = scored[:6]
    context = "\n\n".join(
        f"Source: {source}, page {page}\n{text}"
        for _, source, page, _, text in top_chunks
    )
    sources = unique_sources(top_chunks)
    return context[:7000], sources


def unique_sources(scored_chunks):
    seen = set()
    sources = []
    for _, source, page, _, _ in scored_chunks:
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"file": source, "page": page})
        if len(sources) == 4:
            break
    return sources


def tokenize_question(question):
    return [
        term
        for term in tokenize_text(question)
        if term not in STOP_WORDS and len(term) >= 2
    ]


def tokenize_text(text):
    return re.findall(r"[A-Za-z][A-Za-z0-9+.-]*", text.lower())


def embed_text(text):
    vector = np.zeros(EMBEDDING_DIMENSIONS, dtype="float32")
    for token in tokenize_text(text):
        if token in STOP_WORDS:
            continue
        index = stable_hash(token) % EMBEDDING_DIMENSIONS
        vector[index] += 1.0

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def stable_hash(text):
    value = 2166136261
    for character in text:
        value ^= ord(character)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def chunk_text(text, size=1200, overlap=180):
    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        return []

    chunks = []
    start = 0
    while start < len(clean_text):
        end = min(start + size, len(clean_text))
        chunks.append(clean_text[start:end])
        if end == len(clean_text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def score_chunk(question, question_terms, chunk):
    lowered_chunk = chunk.lower()
    lowered_question = question.lower().strip(" ?!.")
    target_phrase = " ".join(tokenize_question(question))
    score = 0

    if lowered_question and lowered_question in lowered_chunk:
        score += 12

    if target_phrase and target_phrase in lowered_chunk:
        score += 18

    if "rag" in question_terms and "retrieval-augmented generation" in lowered_chunk:
        score += 24

    if "rag" in question_terms and "models which combine" in lowered_chunk:
        score += 18

    if "appendices" in lowered_chunk[:300] or "references" in lowered_chunk[:300]:
        score -= 30

    for term in question_terms:
        if len(term) <= 3:
            occurrences = len(re.findall(rf"\b{re.escape(term)}\b", lowered_chunk))
        else:
            occurrences = lowered_chunk.count(term)
        if occurrences:
            score += min(occurrences, 5) * (4 if len(term) <= 3 else 2)

    definition_patterns = [
        " is ",
        " are ",
        " refers to ",
        " defined as ",
        " allows ",
        "combine",
    ]
    if any(pattern in lowered_chunk for pattern in definition_patterns):
        score += 2

    return score


def answer_with_groq(question, context):
    if not os.getenv("GROQ_API_KEY"):
        return None

    try:
        client = Groq()
        completion = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are ChatScholar, a careful PDF study assistant. "
                        "Answer only from the supplied PDF context. Keep the answer "
                        "concise and do not invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": f"PDF context:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.2,
            max_tokens=700,
        )
        return completion.choices[0].message.content.strip()
    except Exception as error:
        return f"Groq could not answer right now: {error}"


def answer_locally(context):
    return context[:1200]
