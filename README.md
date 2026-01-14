# AI Career Coach – Resume Analysis & RAG Q&A System

An AI-powered web application that analyzes resumes and provides intelligent career insights and resume-aware question answering using Large Language Models (LLMs), vector search, and Retrieval-Augmented Generation (RAG).

---

## 📌 Problem Statement

In today’s competitive job market, candidates often struggle to evaluate how effectively their resume represents their skills, experience, and career trajectory. Traditional resume review processes are:

- Manual and time-consuming  
- Subjective and inconsistent  
- Non-interactive  
- Unable to provide instant, personalized feedback  

Once a resume is reviewed, candidates still lack an easy way to ask follow-up questions such as:
- What are my strongest skills?
- Am I suitable for a particular role?
- What improvements can I make to my resume?

There is a clear need for an **automated, intelligent, and interactive system** that can both analyze resumes and answer questions contextually.

---

## 💡 Solution Overview

This project addresses the problem by building an **AI-powered career coaching system** that:

- Accepts resumes in PDF format
- Extracts and understands resume content using AI
- Generates structured career insights
- Stores resume knowledge semantically using vector embeddings
- Enables resume-aware question answering through Retrieval-Augmented Generation (RAG)

By combining **Large Language Models (LLMs)** with **vector search**, the system produces **grounded, context-aware responses** instead of generic answers.

---

## 🚀 Key Features

- 📄 PDF resume upload
- 🧠 AI-generated resume summary and career insights
- 🔍 Semantic search over resume content
- 💬 Resume-aware conversational Q&A (RAG)
- ⚡ Fast vector retrieval using FAISS
- 🎨 Clean and responsive UI with Tailwind CSS

---

## 🏗️ System Architecture

The application follows a **Retrieval-Augmented Generation (RAG)** architecture to ensure that all responses are grounded in the uploaded resume content.

### High-Level Flow

User
│
│ Upload Resume / Ask Question
▼
Flask Web Application
│
├── PDF Text Extraction (PyPDF)
│
├── Text Chunking
│
├── Embedding Generation
│ (Sentence Transformers)
│
├── Vector Store
│ (FAISS)
│
├── Retriever
│
└── LLM (GPT-4o)
│
▼
Resume Analysis / Context-Aware Answer

css
Copy code

### Architecture Diagram (Mermaid – GitHub Rendered)

```mermaid
flowchart TD
    A[User] --> B[Flask Web App]
    B --> C[Upload Resume]
    C --> D[Text Extraction]
    D --> E[Text Chunking]
    E --> F[Embedding Generation]
    F --> G[FAISS Vector Store]
    G --> H[Retriever]
    H --> I[LLM - GPT-4o]
    I --> J[Resume Analysis / Q&A Response]
🧠 How It Works
1. Resume Upload
The user uploads a PDF resume

Text is extracted from all pages

The text is split into overlapping chunks to preserve semantic context

2. Vector Indexing
Each text chunk is converted into dense embeddings

Embeddings are stored in a FAISS vector database for fast similarity search

3. Resume Analysis
The complete resume text is passed to the LLM

The model generates a structured career summary covering:

Skills

Experience highlights

Education

Career improvement suggestions

4. Resume-Aware Q&A (RAG)
The user submits a question

Relevant resume chunks are retrieved from FAISS

Retrieved context is injected into the LLM prompt

The model generates a grounded, context-aware answer

📂 Project Structure
graphql
Copy code
AI-Career-project/
│
├── templates/
│   ├── index.html        # Resume upload page
│   ├── results.html      # Resume analysis output
│   ├── ask.html          # Ask resume-related questions
│   └── qa_results.html   # Q&A results
│
├── vector_index/         # FAISS vector store (runtime generated)
│   ├── index.faiss
│   └── index.pkl
│
├── app.py                # Flask backend and RAG pipeline
├── requirements.txt      # Python dependencies
🛠️ Tech Stack
Component	Technology
Backend	Flask
LLM	OpenAI GPT-4o
RAG Framework	LangChain
Embeddings	Sentence Transformers
Vector Database	FAISS
PDF Parsing	PyPDF
Frontend	Tailwind CSS

⚙️ Installation & Setup
1. Clone Repository
bash
Copy code
git clone <your-repo-url>
cd AI-Career-project
2. Create Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Environment Variables
Create a .env file:

env
Copy code
OPENAI_API_KEY=your_openai_api_key_here
5. Run the Application
bash
Copy code
python app.py
Open in browser:

cpp
Copy code
http://127.0.0.1:5000
🔐 Security Notes
API keys are managed using environment variables

Uploaded resumes are handled securely

Vector indexes are generated locally at runtime

📌 Use Cases
AI-powered resume evaluation

Career coaching platforms

HR screening and analysis tools

Demonstration of real-world RAG systems

📈 Future Enhancements
Multi-resume support

User authentication and session memory

Persistent chat history

Cloud deployment (Docker / AWS / Render)

Support for additional document formats

🧑‍💻 Author
This project demonstrates practical skills in:

Large Language Model integration

Vector search and semantic retrieval

Retrieval-Augmented Generation (RAG)

Production-style AI application design