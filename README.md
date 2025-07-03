# 🤖 Resume Bot RAG Pipeline

**Resume Bot RAG Pipeline** is a Retrieval-Augmented Generation (RAG) system designed to function as a chatbot that can intelligently answer questions based on resume/portfolio or candidate information documents.

---

## 📘 Overview

The Resume Bot RAG Pipeline aims to provide an intelligent assistant capable of extracting and synthesizing information from a collection of resume and profile related documents.  
By leveraging RAG principles, it combines **document retrieval** with **large language models (LLMs)** to generate accurate and contextually relevant responses to user queries about resumes.

---

## 🧩 Key Components

### **1. `app.py`**
- The main application entry point  
- Launches the chatbot interface with streamlit(CLI or web)

### **2. `rag_pipeline.py`**
Responsible for the core RAG workflow:
- Resume document loading, processing, and chunking
- Generating embeddings for resume content
- Retrieving relevant chunks based on user queries
- Augmenting the LLM’s prompt with retrieved data
- Generating responses using a large language model

### **3. `requirements.txt`**
- Lists all required Python libraries and versions needed to run the project

### **4. `data/`**
- Directory to store raw resume documents or other data to be indexed

### **5. `faiss_index/`**
- Stores the FAISS (Facebook AI Similarity Search) vector index  
- Enables fast similarity search for question answering

### **6. `chat_log.csv`**
- Logs conversational history between the user and the bot  
- Useful for auditing, fine-tuning, or UX improvements

---

## 🛠 Technologies Used

- **Python**: Core programming language
- **FAISS**: For vector similarity search and efficient retrieval
- **Large Language Models (LLMs)**: Used for response generation (e.g. OpenAI GPT, Hugging Face models)
- **LangChain (optional)**: Can be added to streamline RAG components
- **Streamlit / Gradio (optional)**: For a frontend UI

---

## 🚀 Setup & Usage

### **1. Clone the Repository**

```bash
git clone https://github.com/Hemanth-Mydugolam/rag-pipeline-project.git
cd rag-pipeline-project
```

### **2. Install Dependencies**
``` bash
pip install -r requirements.txt
```

### **3. Prepare Data**
Replace all of the existing with your resume documents (.pdf, .csv.) in the data/ directory

### **4. Create FAISS Index**
Replace the resume_docs, portfolio_docs,qna_docs with your data in the function "_build_index" in rag_pipeline.py file and then run the pipeline script to:
- Chunk your documents
- Generate embeddings
- Store them in faiss_index/

```bash
python rag_pipeline.py
```

### **5. Run the Application**
```bash
python app.py
```

#### Start asking resume-related questions like:

- "What are the key skills listed in Hemanth's resume?"
- "Does Hemanth has experience in Python and data analysis?"
