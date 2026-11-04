"""RAG pipeline from notebook/doc.ipynb, packaged for the API.

PDFs in data/ -> chunks -> MiniLM embeddings + BM25 -> reciprocal rank fusion
-> cross-encoder rerank -> Groq LLM answer. Optional query rewriting.
"""
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "data"))
ABSTAIN = "I don't know based on the provided PDFs."
MODES = ["none", "rewrite", "multi_query", "hyde"]


class RagEngine:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        # USE_RERANKER=false saves ~100 MB RAM (useful on 512 MB hosting plans) at some accuracy cost
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") \
            if os.getenv("USE_RERANKER", "true").lower() != "false" else None
        self.llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="openai/gpt-oss-20b",
                            temperature=0.1, max_tokens=1024)
        self._build_index()

    # ---------- indexing ----------
    def _build_index(self):
        pages = []
        for pdf in sorted(DATA_DIR.glob("**/*.pdf")):
            for doc in PyMuPDFLoader(str(pdf)).load():
                doc.metadata["source_file"] = pdf.name
                pages.append(doc)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,
                                                  separators=["\n\n", "\n", " ", ""])
        self.chunks = [{"content": c.page_content, "source": c.metadata["source_file"],
                        "page": c.metadata.get("page", 0)} for c in splitter.split_documents(pages)]
        if not self.chunks:
            raise RuntimeError(f"No PDFs found in {DATA_DIR}")
        texts = [c["content"] for c in self.chunks]
        self.emb = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.tokens = [re.findall(r"\b\w+\b", t.lower()) for t in texts]
        self.counts = [Counter(t) for t in self.tokens]
        self.df = Counter(term for t in self.tokens for term in set(t))
        self.avglen = sum(map(len, self.tokens)) / len(self.tokens)

    # ---------- retrieval ----------
    def _dense(self, text, k):
        q = self.embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return list(np.argsort(-(self.emb @ q))[:k])

    def _bm25(self, text, k, k1=1.5, b=0.75):
        n, terms = len(self.chunks), re.findall(r"\b\w+\b", text.lower())
        scores = np.array([
            sum(math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                * cnt[t] * (k1 + 1) / (cnt[t] + k1 * (1 - b + b * len(tok) / self.avglen))
                for t in terms if t in cnt)
            for tok, cnt in zip(self.tokens, self.counts)])
        return [i for i in np.argsort(-scores)[:k] if scores[i] > 0]

    def retrieve(self, question, search_queries, candidate_k=12, final_k=5):
        fused = Counter()
        for q in search_queries:
            for ranked in (self._dense(q, candidate_k), self._bm25(q, candidate_k)):
                for rank, idx in enumerate(ranked, 1):
                    fused[idx] += 1 / (60 + rank)
        candidates = [i for i, _ in fused.most_common(24)]
        if self.reranker is None:
            return [dict(self.chunks[i], score=round(s, 4)) for i, s in fused.most_common(final_k)]
        scores = self.reranker.predict([(question, self.chunks[i]["content"]) for i in candidates],
                                       show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)[:final_k]
        return [dict(self.chunks[i], score=round(float(s), 3)) for i, s in ranked]

    # ---------- query rewriting ----------
    def _llm(self, prompt):
        return self.llm.invoke(prompt).content.strip()

    def search_queries(self, question, mode, history):
        convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
        ctx = f"Conversation so far:\n{convo}\n\n" if convo else ""
        if mode == "rewrite":
            q = self._llm(f"{ctx}Rewrite the user's latest question into one standalone search query for an AI "
                          f"textbook. Fix spelling, expand abbreviations, name exact concepts. Return only the query.\n"
                          f"Question: {question}\nSearch query:").strip('"')
            return [q or question]
        if mode == "multi_query":
            out = self._llm(f"{ctx}Write 3 different search queries that would retrieve passages answering this "
                            f"question from an AI textbook. One per line, no numbering.\nQuestion: {question}")
            lines = [re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", l).strip() for l in out.splitlines()]
            return [question] + [l for l in lines if l][:3]
        if mode == "hyde":
            return [question, self._llm(f"{ctx}Write a short factual textbook paragraph (3-4 sentences) "
                                        f"answering: {question}")]
        return [question]

    # ---------- full pipeline ----------
    def answer(self, question, mode="none", history=()):
        t0 = time.perf_counter()
        queries = self.search_queries(question, mode, list(history))
        docs = self.retrieve(question, queries)
        t1 = time.perf_counter()
        context = "\n\n---\n\n".join(f"Source {i + 1}: {d['content']}" for i, d in enumerate(docs))
        answer = self._llm(f"Answer only from this context. If absent, say: {ABSTAIN} Cite [Source n].\n\n"
                           f"{context}\n\nQuestion: {question}\nAnswer:") if docs else ABSTAIN
        t2 = time.perf_counter()
        return {"answer": answer, "search_queries": queries,
                "sources": [{"source": d["source"], "page": d["page"], "score": d["score"],
                             "preview": d["content"][:200]} for d in docs],
                "retrieval_ms": round((t1 - t0) * 1000), "generation_ms": round((t2 - t1) * 1000)}
