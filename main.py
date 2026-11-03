"""Small PDF RAG application using Sentence Transformers, ChromaDB, and Groq.

Usage:
    1. Create a .env file containing GROQ_API_KEY=your_key
    2. Put PDF files in data/
    3. Run: python main.py
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vectordb"
COLLECTION_NAME = "pdf_docs"


class EmbeddingManager:
    """Loads a local embedding model and converts text into vectors."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded. Vector dimension: {dimension}")

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("No texts were provided for embedding.")
        print(f"Generating embeddings for {len(texts)} chunks...")
        return self.model.encode(texts, show_progress_bar=True)


class VectorStore:
    """Stores PDF chunks and their vectors in a persistent ChromaDB collection."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_path: Path = VECTOR_DB_DIR,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = Path(persist_path)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Embeddings of PDF document chunks"},
        )
        print(
            f"ChromaDB ready: '{self.collection_name}' "
            f"({self.collection.count()} chunks stored)"
        )

    def add_documents(self, documents: list[Document], embeddings: np.ndarray) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("The number of documents and embeddings must match.")
        if not documents:
            raise ValueError("No document chunks were provided.")

        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        texts: list[str] = []
        vectors: list[list[float]] = []

        for index, (document, embedding) in enumerate(zip(documents, embeddings)):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = index
            metadata["content_length"] = len(document.page_content)
            ids.append(f"chunk_{uuid.uuid4().hex}")
            metadatas.append(metadata)
            texts.append(document.page_content)
            vectors.append(embedding.tolist())

        # Chroma expects a 2D list: one embedding vector per document.
        self.collection.add(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
            documents=texts,
        )
        print(f"Stored {len(documents)} chunks. Total: {self.collection.count()}")

    def similarity_search(
        self, query_embedding: np.ndarray, n_results: int = 4
    ) -> list[Document]:
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, max(self.collection.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            Document(
                page_content=text,
                metadata={**(metadata or {}), "distance": distance},
            )
            for text, metadata, distance in zip(documents, metadatas, distances)
        ]


def load_and_split_pdfs(data_directory: Path) -> list[Document]:
    """Read every PDF in data/ and split it into overlapping chunks."""
    pdf_files = list(data_directory.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {data_directory}")

    pages: list[Document] = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        pages.extend(PyPDFLoader(str(pdf_path)).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    print(f"Loaded {len(pages)} pages and created {len(chunks)} chunks.")
    return chunks


def create_llm() -> ChatGroq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")
    return ChatGroq(
        groq_api_key=api_key,
        model_name="openai/gpt-oss-20b",
        temperature=0.1,
        max_tokens=1024,
    )


def rag(query: str, llm: ChatGroq, store: VectorStore, embedder: EmbeddingManager) -> str:
    """Retrieve relevant chunks and ask the LLM to answer only from them."""
    query_vector = embedder.generate_embeddings([query])[0]
    relevant_chunks = store.similarity_search(query_vector, n_results=4)
    if not relevant_chunks:
        return "I could not find relevant content in the stored PDFs."

    context = "\n\n---\n\n".join(chunk.page_content for chunk in relevant_chunks)
    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say: "I don't know based on the provided PDFs."
Keep the answer concise and clear.

Context:
{context}

Question: {query}
Answer:"""
    return llm.invoke(prompt).content


def main() -> None:
    embedder = EmbeddingManager()
    store = VectorStore()

    # Index PDFs only when the collection is empty. Delete data/vectordb to rebuild.
    if store.collection.count() == 0:
        chunks = load_and_split_pdfs(DATA_DIR)
        embeddings = embedder.generate_embeddings(
            [chunk.page_content for chunk in chunks]
        )
        store.add_documents(chunks, embeddings)

    llm = create_llm()
    print("\nPDF RAG is ready. Type 'exit' to quit.")
    while True:
        question = input("\nAsk a question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print("\nAnswer:", rag(question, llm, store, embedder))


if __name__ == "__main__":
    main()
