"""RAG pipeline from notebook/doc.ipynb, packaged for the API.

PDFs in data/ -> chunks -> MiniLM embeddings + BM25 -> reciprocal rank fusion
-> cross-encoder rerank -> Groq LLM answer. Optional query rewriting.
"""
import hashlib
import json
import math
import os
import random
import threading
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
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE, CHUNK_OVERLAP = 1000, 200
# Chunks + embeddings are saved here so a restart doesn't re-read and re-embed every PDF.
# The Docker image builds it at build time; it is rebuilt automatically when the PDFs change.
INDEX_CACHE = Path(os.getenv("RAG_INDEX_CACHE", DATA_DIR / ".index_cache.npz"))


class RagEngine:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        # USE_RERANKER=false saves ~100 MB RAM (useful on 512 MB hosting plans) at some accuracy cost
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") \
            if os.getenv("USE_RERANKER", "true").lower() != "false" else None
        self.llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="openai/gpt-oss-20b",
                            temperature=0.1, max_tokens=1024)
        self._suggestions = None
        self._suggest_lock = threading.Lock()
        self._build_index()

    # ---------- indexing ----------
    def _corpus_key(self, pdfs):
        sig = [EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP] + [[str(p.relative_to(DATA_DIR)), p.stat().st_size] for p in pdfs]
        return hashlib.sha256(json.dumps(sig).encode()).hexdigest()

    def _build_index(self):
        pdfs = sorted(DATA_DIR.glob("**/*.pdf"))
        key = self._corpus_key(pdfs)
        if not self._load_cache(key):
            self._index_pdfs(pdfs)
            self._save_cache(key)
        texts = [c["content"] for c in self.chunks]
        self.tokens = [re.findall(r"\b\w+\b", t.lower()) for t in texts]
        self.counts = [Counter(t) for t in self.tokens]
        self.df = Counter(term for t in self.tokens for term in set(t))
        self.avglen = sum(map(len, self.tokens)) / len(self.tokens)

    def _index_pdfs(self, pdfs):
        pages, self.documents = [], []
        for pdf in pdfs:
            loaded = PyMuPDFLoader(str(pdf)).load()
            for doc in loaded:
                doc.metadata["source_file"] = pdf.name
                pages.append(doc)
            self.documents.append({"name": pdf.name, "pages": len(loaded)})
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                                  separators=["\n\n", "\n", " ", ""])
        self.chunks = [{"content": c.page_content, "source": c.metadata["source_file"],
                        "page": c.metadata.get("page", 0)} for c in splitter.split_documents(pages)]
        if not self.chunks:
            raise RuntimeError(f"No PDFs found in {DATA_DIR}")
        per_file = Counter(c["source"] for c in self.chunks)
        for d in self.documents:
            d["chunks"] = per_file[d["name"]]
        self.emb = self.embedder.encode([c["content"] for c in self.chunks], batch_size=64,
                                        normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    def _load_cache(self, key):
        try:
            with np.load(INDEX_CACHE, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                if meta["key"] != key:
                    return False
                self.chunks, self.documents, self.emb = meta["chunks"], meta["documents"], z["emb"]
            print(f"Loaded index cache: {len(self.chunks)} chunks from {INDEX_CACHE}", flush=True)
            return True
        except (OSError, KeyError, ValueError):
            return False

    def _save_cache(self, key):
        try:
            INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
            meta = json.dumps({"key": key, "chunks": self.chunks, "documents": self.documents})
            tmp = INDEX_CACHE.with_name(INDEX_CACHE.stem + ".tmp.npz")
            np.savez(tmp, emb=self.emb, meta=np.array(meta))
            tmp.replace(INDEX_CACHE)
        except OSError as e:                     # read-only disk etc.: just run without a cache
            print(f"Could not write index cache: {e}", flush=True)

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

    # ---------- starter questions, written by the LLM from the indexed PDFs ----------
    def suggestions(self, n=4):
        """n example questions, each answerable from a passage of a different PDF. Cached after first success."""
        with self._suggest_lock:
            if self._suggestions is None:
                rng = random.Random()
                files = [d["name"] for d in self.documents if d["chunks"]]
                picked = rng.sample(files, min(n, len(files)))
                while len(picked) < n and files:            # fewer PDFs than n: reuse some
                    picked.append(rng.choice(files))
                passages = []
                for name in picked:
                    idx = [i for i, c in enumerate(self.chunks) if c["source"] == name]
                    middle = idx[len(idx) // 5: max(len(idx) // 5 + 1, 4 * len(idx) // 5)]   # skip cover pages/indexes
                    passages.append(self.chunks[rng.choice(middle)]["content"][:900])
                out = self._llm("For each passage below, write one short, natural question a reader might ask that "
                                "the passage answers. Don't mention 'the passage'. One question per line, "
                                "no numbering, nothing else.\n\n"
                                + "\n\n".join(f"Passage {i + 1}:\n{p}" for i, p in enumerate(passages)))
                qs = [re.sub(r"^\s*(?:\d+[.)]|[-*•]|Passage \d+:)\s*", "", l).strip() for l in out.splitlines()]
                qs = [q for q in qs if q.endswith("?")][:n]
                if qs:
                    self._suggestions = qs
            return self._suggestions or []

    # ---------- query rewriting ----------
    def _llm(self, prompt):
        return self.llm.invoke(prompt).content.strip()

    def search_queries(self, question, mode, history):
        convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
        ctx = f"Conversation so far:\n{convo}\n\n" if convo else ""
        if mode == "rewrite":
            q = self._llm(f"{ctx}Rewrite the user's latest question into one standalone search query for a collection of "
                          f"documents (acts, rules, regulations, reports). Fix spelling, expand abbreviations, use the "
                          f"exact legal or technical terms. Return only the query.\n"
                          f"Question: {question}\nSearch query:").strip('"')
            return [q or question]
        if mode == "multi_query":
            out = self._llm(f"{ctx}Write 3 different search queries that would retrieve passages answering this "
                            f"question from a collection of documents (acts, rules, regulations, reports). One per line, no numbering.\nQuestion: {question}")
            lines = [re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", l).strip() for l in out.splitlines()]
            return [question] + [l for l in lines if l][:3]
        if mode == "hyde":
            return [question, self._llm(f"{ctx}Write a short factual paragraph (3-4 sentences), in the "
                                        f"style of the source document, answering: {question}")]
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
