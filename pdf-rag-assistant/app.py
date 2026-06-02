from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from essay_grader import grade_essay
from rag import build_document_store, answer_question

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "data"
app.config["FAISS_FOLDER"] = "faiss_index"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

document_store = build_document_store(
    app.config["UPLOAD_FOLDER"],
    app.config["FAISS_FOLDER"],
)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    upload_message = None

    if request.method == "POST":
        files = [file for file in request.files.getlist("pdf") if file.filename]
        uploaded = []

        if not files:
            upload_message = "Choose a PDF first."
        else:
            for file in files:
                if not file.filename.lower().endswith(".pdf"):
                    continue

                filename = secure_filename(file.filename)
                path = f"{app.config['UPLOAD_FOLDER']}/{filename}"
                file.save(path)
                uploaded.append(filename)

            if uploaded:
                document_store.sync()
                upload_message = f"Uploaded {len(uploaded)} PDF file(s)."
            else:
                upload_message = "Only PDF files are supported."

    return render_template(
        "chat.html",
        files=document_store.files(),
        upload_message=upload_message,
    )


@app.route("/essay", methods=["GET", "POST"])
def essay():
    result = None
    essay_text = ""

    if request.method == "POST":
        essay_text = request.form.get("essay", "").strip()
        result = grade_essay(essay_text)

    return render_template("essay.html", result=result, essay_text=essay_text)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Ask a question first."}), 400

    result = answer_question(question, document_store)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "pdf_count": len(document_store.files()),
            "chunk_count": document_store.chunk_count(),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
