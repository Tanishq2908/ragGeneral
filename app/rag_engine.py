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
from concurrent.futures import ThreadPoolExecutor
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
STOPWORDS = frozenset("""a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here
hers him his how i if in into is it its itself just me more most my no nor not now of off on once only or other our out
over own same she should so some such than that the their them then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you your shall may such any also see
s t don per within upon thereof therein""".split())


def is_abstain(text):
    """True when the model said it couldn't find the answer (also catches variants like "I do not know")."""
    t = re.sub(r"\s+", " ", (text or "").strip().lower()).replace("’", "'")
    return t.startswith(("i don't know", "i do not know", "i dont know")) or t == ""


def clean_answer(text):
    """An abstention is normalised to the exact ABSTAIN sentence (drops stray '[Source 1]' etc.)."""
    return ABSTAIN if is_abstain(text) else text.strip()


def fallback_modes(mode):
    """Modes to try in order: the chosen one, then a broader search if it found nothing."""
    order = [mode]
    for m in ("multi_query", "hyde"):
        if m not in order:
            order.append(m)
            break
    return order
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
                            temperature=0.1, max_tokens=2048)
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
    def _dense_scores(self, text):
        q = self.embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return self.emb @ q                                   # cosine similarity (embeddings are normalised)

    def _bm25_scores(self, text, k1=1.5, b=0.75):
        n, terms = len(self.chunks), re.findall(r"\b\w+\b", text.lower())
        return np.array([
            sum(math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                * cnt[t] * (k1 + 1) / (cnt[t] + k1 * (1 - b + b * len(tok) / self.avglen))
                for t in terms if t in cnt)
            for tok, cnt in zip(self.tokens, self.counts)])

    def _dense(self, text, k):
        return list(np.argsort(-self._dense_scores(text))[:k])

    def _bm25(self, text, k):
        scores = self._bm25_scores(text)
        return [i for i in np.argsort(-scores)[:k] if scores[i] > 0]

    # ---------- lexical NLP measures ----------
    def _idf(self, term):
        return math.log((len(self.chunks) + 1) / (self.df.get(term, 0) + 1)) + 1

    def _tfidf(self, tokens):
        counts = Counter(t for t in tokens if t not in STOPWORDS)
        total = sum(counts.values()) or 1
        return {t: c / total * self._idf(t) for t, c in counts.items()}

    @staticmethod
    def _cosine(a, b):
        dot = sum(v * b.get(t, 0.0) for t, v in a.items())
        na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    @staticmethod
    def _content_words(text):
        return [t for t in re.findall(r"\b\w+\b", text.lower()) if t not in STOPWORDS and not t.isdigit()]

    # ---------- document awareness ----------
    _GENERIC = frozenset("act acts rules rule regulations regulation the of and as amended official text gazette india "
                         "e amendment 2023 en pdf".split())

    def _doc_terms(self, name):
        words = re.findall(r"[a-z]+", Path(name).stem.lower().replace("_", " "))
        terms = {w.rstrip("s") for w in words if w not in self._GENERIC and len(w) > 2}
        kind = "rules" if re.search(r"\brules?\b", name.lower()) else \
               "regulations" if "regulation" in name.lower() else \
               "amendment" if "amendment" in name.lower() else "act"
        return terms, kind

    def docs_mentioned(self, text):
        """Documents a question names, e.g. 'Patents Act' -> patents-act-1970.pdf (not the Patent Rules)."""
        t = text.lower()
        words = {w.rstrip("s") for w in re.findall(r"[a-z]+", t)}
        says = {"rules": bool(re.search(r"\brules?\b", t)), "regulations": "regulation" in t,
                "amendment": "amendment" in t, "act": bool(re.search(r"\bacts?\b", t))}
        hits = set()
        for d in self.documents:
            terms, kind = self._doc_terms(d["name"])
            if not terms:
                continue
            found = len(terms & words)
            enough = found == len(terms) or (found >= 2 and found / len(terms) >= 0.6)
            if enough and (says[kind] or not any(says.values())):
                hits.add(d["name"])
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", t))
        with_year = {h for h in hits if any(y in h for y in years)}
        return with_year or hits          # "Patent Rules 2024" -> only the 2024 Rules

    def _allowed_mask(self, sources):
        if not sources:
            return None
        return np.array([c["source"] in sources for c in self.chunks])

    def retrieve(self, question, search_queries, candidate_k=20, final_k=5, sources=None):
        """Hybrid retrieval. Each returned chunk carries `score` (the ranking score) and `scores`, a breakdown
        of NLP measures between the question and the chunk. `sources` limits the search to those PDFs."""
        mask = self._allowed_mask(sources)
        fused = Counter()
        for q in search_queries:
            dense, bm25 = self._dense_scores(q), self._bm25_scores(q)
            if mask is not None:
                dense = np.where(mask, dense, -np.inf)
                bm25 = np.where(mask, bm25, 0.0)
            dense_ranked = list(np.argsort(-dense)[:candidate_k])
            bm25_ranked = [i for i in np.argsort(-bm25)[:candidate_k] if bm25[i] > 0]
            for ranked in (dense_ranked, bm25_ranked):
                for rank, idx in enumerate(ranked, 1):
                    fused[idx] += 1 / (60 + rank)
        candidates = [i for i, _ in fused.most_common(24)]
        rerank = {}
        if self.reranker is None:
            ranked = [(i, s) for i, s in fused.most_common(final_k)]
        else:
            preds = self.reranker.predict([(question, self.chunks[i]["content"]) for i in candidates],
                                          show_progress_bar=False)
            rerank = {i: float(p) for i, p in zip(candidates, preds)}
            ranked = sorted(rerank.items(), key=lambda x: x[1], reverse=True)[:final_k]

        dense, bm25 = self._dense_scores(question), self._bm25_scores(question)
        dense_rank = np.empty(len(dense), int); dense_rank[np.argsort(-dense)] = np.arange(1, len(dense) + 1)
        bm25_rank = np.empty(len(bm25), int); bm25_rank[np.argsort(-bm25)] = np.arange(1, len(bm25) + 1)
        q_words = set(self._content_words(question))
        q_vec = self._tfidf(re.findall(r"\b\w+\b", question.lower()))
        out = []
        for i, s in ranked:
            c_words = set(self._content_words(self.chunks[i]["content"]))
            scores = {
                "dense_cosine": round(float(dense[i]), 4),
                "bm25": round(float(bm25[i]), 3),
                "tfidf_cosine": round(self._cosine(q_vec, self._tfidf(self.tokens[i])), 4),
                "keyword_jaccard": round(len(q_words & c_words) / len(q_words | c_words), 4) if q_words | c_words else 0.0,
                "query_term_coverage": round(len(q_words & c_words) / len(q_words), 3) if q_words else 0.0,
                "rrf_fusion": round(fused[i], 4),
                "dense_rank": int(dense_rank[i]),
                "bm25_rank": int(bm25_rank[i]),
                "matched_terms": sorted(q_words & c_words, key=lambda t: -self._idf(t))[:8],
            }
            if i in rerank:
                scores["rerank"] = round(rerank[i], 3)
            out.append(dict(self.chunks[i], score=round(float(s), 4 if self.reranker is None else 3), scores=scores))
        return out

    def analyze(self, question, answer, docs):
        """NLP measures of an answer: semantic relevance/groundedness, lexical support, citations, lexical diversity."""
        cited = sorted({int(n) for n in re.findall(r"Source\s*(\d+)", answer)})
        answer = re.sub(r"\[?\(?Sources?\s*[\d,\s]+\]?\)?", " ", answer)        # drop citation markers for lexical stats
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", answer)
        sentences = max(1, len(re.findall(r"[.!?]+(?:\s|$)", answer)))
        n_words = len(words) or 1
        content = self._content_words(answer)
        context = " ".join(d["content"] for d in docs)
        ctx_words = set(self._content_words(context))
        a_vec = self._tfidf(re.findall(r"\b\w+\b", answer.lower()))
        result = {
            "abstained": is_abstain(answer),
            "word_count": len(words),
            "sentence_count": sentences,
            "lexical_diversity": round(len({w.lower() for w in words}) / n_words, 3),
            "lexical_support": round(sum(t in ctx_words for t in content) / len(content), 3) if content else 0.0,
            "tfidf_answer_context": round(self._cosine(a_vec, self._tfidf(re.findall(r"\b\w+\b", context.lower()))), 4),
            "citations": cited,
            "valid_citations": [c for c in cited if 1 <= c <= len(docs)],
            "top_keywords": [t for t, _ in sorted(a_vec.items(), key=lambda x: -x[1]) if not t.isdigit() and len(t) > 2][:6],
        }
        texts = [question, answer] + [d["content"] for d in docs]
        e = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        result["answer_relevance"] = round(float(e[0] @ e[1]), 4)                      # answer vs question
        if docs:
            sims = e[2:] @ e[1]
            result["groundedness_max"] = round(float(sims.max()), 4)                  # answer vs best source
            result["groundedness_mean"] = round(float(sims.mean()), 4)
        return result

    # ---------- starter questions, written by the LLM from the indexed PDFs ----------
    @staticmethod
    def _is_prose(text):
        """True for running text; False for fee tables, form lists, indexes and other number-heavy chunks."""
        letters = sum(ch.isalpha() for ch in text)
        digits = sum(ch.isdigit() for ch in text)
        words = re.findall(r"[A-Za-z]{3,}", text)
        return len(text) >= 500 and letters / max(1, len(text)) > 0.68 and digits / max(1, len(text)) < 0.04 \
            and len(words) >= 60

    def _candidate_questions(self, rng, n):
        files = [d["name"] for d in self.documents if d["chunks"]]
        picked = rng.sample(files, min(n, len(files)))
        while len(picked) < n and files:
            picked.append(rng.choice(files))
        passages = []
        for name in picked:
            idx = [i for i, c in enumerate(self.chunks) if c["source"] == name and self._is_prose(c["content"])]
            if not idx:
                continue
            passages.append((name, self.chunks[rng.choice(idx)]["content"][:900]))
        if not passages:
            return []
        out = self._llm(
            "You write example questions for a search app over Indian laws and regulations.\n"
            "For each passage below, write ONE question that:\n"
            "- is answered clearly and completely by that passage alone;\n"
            "- is self-contained: name the Act, Rules or Regulations it is about (use the document title), and never "
            "refer to a bare section/rule/form number without saying which law it belongs to;\n"
            "- is short (under 15 words) and natural, like something a student or lawyer would ask;\n"
            "- does not mention 'the passage' or 'the text'.\n"
            "Return only the questions, one per line, no numbering.\n\n"
            + "\n\n".join(f"Passage {i + 1} (from {Path(n_).stem}):\n{p}" for i, (n_, p) in enumerate(passages)))
        qs = [re.sub(r"^\s*(?:\d+[.)]|[-*•]|Passage \d+[:.)]?)\s*", "", l).strip() for l in out.splitlines()]
        return [q for q in qs if q.endswith("?") and 4 <= len(q.split()) <= 25]

    def answerable(self, question):
        """Run the full pipeline once and report whether it produces a real (non-abstaining) answer."""
        try:
            docs = self.prepare(question, "none")["docs"]
            return bool(docs) and not is_abstain(self._llm(self._answer_prompt(question, docs)))
        except Exception:
            return False

    def suggestions(self, n=4):
        """n example questions from different PDFs, each checked by actually answering it. Cached after first success."""
        with self._suggest_lock:
            if self._suggestions is None:
                rng, verified = random.Random(), []
                for _ in range(2):                                   # at most two rounds of candidates
                    candidates = [q for q in self._candidate_questions(rng, 2 * n) if q not in verified]
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        ok = list(pool.map(self.answerable, candidates))
                    verified += [q for q, good in zip(candidates, ok) if good]
                    if len(verified) >= n:
                        break
                if verified:
                    self._suggestions = verified[:n]
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
    def is_complex(self, question):
        """Multi-part questions: long, several provisions/sections, or several laws named."""
        sections = re.findall(r"\b\d+[A-Z]?\s*\([a-z0-9]{1,4}\)|\b(?:section|sec\.|rule)\s+\d+", question, re.I)
        return (len(question.split()) >= 22 or len(sections) >= 2 or len(self.docs_mentioned(question)) >= 2
                or question.count("?") >= 2)

    def decompose(self, question, history=()):
        """Split a complex question into 2-4 focused, self-contained search questions (one LLM call)."""
        if not self.is_complex(question):
            return []
        titles = "; ".join(Path(d["name"]).stem for d in self.documents)
        convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in list(history)[-4:])
        out = self._llm(
            (f"Conversation so far:\n{convo}\n\n" if convo else "")
            + "Split the user's question into 2 to 4 short search questions for a library of Indian laws "
            f"({titles}). Each search question must cover ONE law or provision, name that Act/Rules/Regulations "
            "explicitly, keep section and clause numbers exactly as written (e.g. Section 3(d)), and add the key "
            "legal terms someone would find in that provision. Return only the questions, one per line.\n\n"
            f"Question: {question}")
        qs = [re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", l).strip() for l in out.splitlines()]
        return [q for q in qs if len(q.split()) >= 3][:4]

    @staticmethod
    def _merge(lists, k):
        """Round-robin merge of ranked chunk lists, dropping duplicates, so every part of a question is covered."""
        seen, out = set(), []
        for rank in range(max((len(l) for l in lists), default=0)):
            for l in lists:
                if rank < len(l):
                    key = (l[rank]["source"], l[rank]["page"], l[rank]["content"][:80])
                    if key not in seen:
                        seen.add(key)
                        out.append(l[rank])
        return out[:k]

    def prepare(self, question, mode="none", history=()):
        """Plan + retrieve. Complex questions are decomposed and each part is searched (within the PDFs it names);
        the results are merged. Returns search queries, sub-questions, retrieved chunks and the time taken."""
        t0 = time.perf_counter()
        queries = self.search_queries(question, mode, list(history))
        parts = self.decompose(question, history)
        main = self.retrieve(question, queries, final_k=5 if not parts else 4, sources=None)
        named = self.docs_mentioned(question)
        lists = [main]
        if parts:
            for part in parts:
                lists.append(self.retrieve(part, [part], final_k=3, sources=self.docs_mentioned(part) or None))
            docs = self._merge(lists, 8)
        elif named and not any(d["source"] in named for d in main):
            # the question names a law but the open search missed it: add the best passages from that law
            docs = self._merge([main, self.retrieve(question, queries, final_k=2, sources=named)], 6)
        else:
            docs = main
        return {"search_queries": queries, "sub_questions": parts, "docs": docs,
                "retrieval_ms": round((time.perf_counter() - t0) * 1000)}

    @staticmethod
    def _answer_prompt(question, docs, parts=()):
        context = "\n\n---\n\n".join(f"[Source {i + 1}] {Path(d['source']).stem}, page {d['page'] + 1}:\n{d['content']}"
                                       for i, d in enumerate(docs))
        plan = ("\nThe question has these parts; answer each one, in this order:\n"
                + "\n".join(f"{i}. {p}" for i, p in enumerate(parts, 1)) + "\n") if parts else ""
        return (
            "You are a careful legal research assistant for Indian law. Use ONLY the numbered sources below, which "
            "are extracts from Indian Acts, Rules and Regulations.\n\n"
            "Format your reply in Markdown like this:\n"
            "**Answer:** a direct answer in 1-3 sentences.\n\n"
            "**Explanation**\n"
            "- Short bullet points (or numbered steps for a procedure or a step-by-step analysis). For a question with "
            "several parts, give each part its own short heading (### Part name) in the order asked.\n"
            "- For every legal point, name the Act/Rules and the section, rule or clause, and cite the source right "
            "after the statement, like [Source 2].\n"
            "- When exact wording matters, quote the key phrase from the law in quotation marks.\n\n"
            "**Key provisions:** one line listing the provisions used, e.g. Patents Act 1970, s.3(d) [Source 1].\n\n"
            "Rules:\n"
            "- For scenario or \"how would you determine\" questions, apply the provisions in the sources to the facts "
            "step by step. You may reason from the sources, but do not add legal rules that are not in them.\n"
            "- If a source is a table (fees, forms, schedules), read the row that matches the question.\n"
            "- If the sources cover only some parts, answer those parts and finish with **Not covered by the "
            "documents:** listing what is missing. Never refuse the whole question because one part is missing.\n"
            "- Plain, precise language; usually under 350 words.\n"
            f"- Only if NONE of the sources is relevant to any part of the question, reply exactly: {ABSTAIN}\n"
            f"{plan}\n"
            f"{context}\n\nQuestion: {question}\nReply:")

    @staticmethod
    def sources(docs):
        return [{"source": d["source"], "page": d["page"], "score": d["score"], "preview": d["content"][:200],
                 "scores": d.get("scores", {})} for d in docs]

    def stream_answer(self, question, docs, parts=()):
        """Yields the answer text piece by piece as the LLM writes it."""
        if not docs:
            yield ABSTAIN
            return
        for chunk in self.llm.stream(self._answer_prompt(question, docs, parts)):
            if chunk.content:
                yield chunk.content

    def answer(self, question, mode="none", history=()):
        for attempt in fallback_modes(mode):
            prep = self.prepare(question, attempt, history)
            docs = prep["docs"]
            t1 = time.perf_counter()
            answer = self._llm(self._answer_prompt(question, docs, prep["sub_questions"])) if docs else ABSTAIN
            if not is_abstain(answer):
                break
        answer = clean_answer(answer)
        return {"answer": answer, "search_queries": prep["search_queries"], "sub_questions": prep["sub_questions"],
                "sources": self.sources(docs), "_docs": docs,
                "retrieval_ms": prep["retrieval_ms"], "generation_ms": round((time.perf_counter() - t1) * 1000)}
