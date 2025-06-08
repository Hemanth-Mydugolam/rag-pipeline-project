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

class RAGPipeline:
    def __init__(self, temperature: float = 0.3, model: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(temperature=temperature, model=model)
        self.index_path = "faiss_index"  # directory or file prefix for saved FAISS index
        self.index = VectorIndex()
        self.loader = DocumentLoader()
        self.qa_chain: RetrievalQA | None = None

        # Always try loading the FAISS index from saved file
        if os.path.exists(self.index_path):
            print("Loading FAISS index from disk...")
            self.index.load(self.index_path)
            self._initialize_chain()
        else:
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Please build the index first outside this code."
            )

    # Optional: method to rebuild index if needed, NOT called in __init__
    def build_index_from_sources(self, resume_path: str, portfolio_url: str, csv_path: str):
        resume_docs = self.loader.load(resume_path)
        portfolio_docs = self.loader.load(portfolio_url)
        qna_docs = self.loader.load_csv_as_docs(csv_path)
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

        response = self.qa_chain.invoke({"query": question})

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
