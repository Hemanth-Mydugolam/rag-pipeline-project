import os
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please set it in .env or environment variables.")


class DocumentLoader:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def load(self, source: Union[str, Path]) -> List[Document]:
        src = str(source).lower()
        if src.startswith("http"):
            raw_docs = WebBaseLoader(source).load()
        elif src.endswith(".pdf"):
            raw_docs = PyPDFLoader(source).load()
        else:
            raise ValueError(f"Unsupported source type: {source}")
        return self.splitter.split_documents(raw_docs)

    def load_csv_as_docs(self, csv_path: str) -> List[Document]:
        df = pd.read_csv(csv_path)
        documents = []
        for _, row in df.iterrows():
            q = str(row[0]).strip()
            a = str(row[1]).strip()
            content = f"Q: {q}\nA: {a}"
            documents.append(Document(page_content=content, metadata={"source": csv_path}))
        return documents  # Do not split QA pairs


class VectorIndex:
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.store: FAISS | None = None

    def build(self, documents: List[Document], index_path: str):
        self.store = FAISS.from_documents(documents, self.embeddings)
        self.store.save_local(index_path)

    def load(self, index_path: str):
        self.store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        if self.store is None:
            raise ValueError("Index not loaded")
        return self.store.similarity_search_with_score(query, k=top_k)


class RAGPipeline:
    def __init__(self, temperature: float = 0.3, model: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(temperature=temperature, model=model)
        self.index_path = "faiss_index"
        self.index = VectorIndex()
        self.loader = DocumentLoader()
        self.qa_chain: RetrievalQA | None = None

        # Load index if it exists, otherwise build it
        if os.path.exists(self.index_path):
            self.index.load(self.index_path)
            self._initialize_chain()
        else:
            self._build_index()

    def _build_index(self):
        resume_docs = self.loader.load(".data/Hemanth_Mydugolam_Resume.pdf")
        portfolio_docs = self.loader.load("https://hemanth-mydugolam.shinyapps.io/Portfolio/")
        qna_docs = self.loader.load_csv_as_docs("./data/Aboutme-HemanthMydugolam.csv")
        all_docs = resume_docs + portfolio_docs + qna_docs
        self.index.build(all_docs, self.index_path)
        self._initialize_chain()
    

    def _initialize_chain(self):
        system_prompt = (
            "You are ResumeBot, a helpful and polite assistant that helps users learn more about Hemanth Mydugolam's "
            "resume, skills, experience, and qualifications. Use only the provided documents and context to answer. "
            "If the user asks something outside of this, politely decline. If the question involves areas like Machine Learning, "
            "Database Management, or Computer Science, you can creatively infer based on Hemanth’s known skills, but stay grounded."
        )

        full_prompt = (
            f"<|system|>\n{system_prompt}\n"
            "<|user|>\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n"
        )

        prompt = PromptTemplate(
            template=full_prompt,
            input_variables=["context", "question"],
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.index.store.as_retriever(search_type="similarity"),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )



    def query(self, question: str, top_k: int = 4) -> dict:
        if self.qa_chain is None:
            raise RuntimeError("QA chain not initialized.")

        # Run the retrieval QA chain to get answer and source documents
        #response = self.qa_chain({"query": question})
        response = self.qa_chain.invoke({"query": question})

        # Retrieve documents with scores separately
        scored = self.index.search(question, top_k=top_k)

        results = [
            (
                Path(d.metadata.get("source", "unknown")).name,
                d.metadata.get("page", "-"),
                round(score, 4),
            )
            for d, score in scored
        ]

        return {"answer": response["result"], "results": results}
