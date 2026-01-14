import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from pypdf import PdfReader

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# =========================
# ENV SETUP
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
VECTOR_STORE = "vector_index"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# AI COMPONENTS
# =========================
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

# =========================
# PROMPT
# =========================
resume_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
You are an AI Career Coach.

Analyze the resume below and provide:
- Career Summary
- Key Skills
- Experience Highlights
- Education
- Improvement Suggestions

Resume:
{resume}
"""
)

# =========================
# HELPERS
# =========================
def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def build_vector_store(text: str):
    chunks = text_splitter.split_text(text)
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local(VECTOR_STORE)


def load_vector_store():
    return FAISS.load_local(
        VECTOR_STORE,
        embeddings,
        allow_dangerous_deserialization=True
    )


# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_resume():
    file = request.files.get("file")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    resume_text = extract_text_from_pdf(path)

    build_vector_store(resume_text)

    summary = llm.invoke(
        resume_prompt.format(resume=resume_text)
    ).content

    return render_template("results.html", resume_analysis=summary)


@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.method == "POST":
        query = request.form["query"]

        db = load_vector_store()
        retriever = db.as_retriever(search_kwargs={"k": 4})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

        answer = qa_chain.run(query)

        return render_template(
            "qa_results.html",
            query=query,
            result=answer
        )

    return render_template("ask.html")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
