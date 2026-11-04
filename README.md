# ragGeneral

Hybrid RAG over the PDFs in `data/`: MiniLM dense search + BM25, reciprocal rank fusion, cross-encoder reranking, and a Groq LLM to write the answer.

## Notebook: `notebook/doc.ipynb`
The last cells evaluate the pipeline on `data/eval_questions.json`, which has 201 labelled questions about `AIcorpus.pdf`:

| type | count | what it tests |
|---|---|---|
| single | 133 | one-section factual lookups (includes the PDF's gold set G01–G14) |
| multi | 23 | evidence from 2+ sections |
| conversational | 25 | casual or vague phrasing, where query rewriting should help |
| distractor | 4 | near-duplicate vocabulary |
| unanswerable | 16 | the system must abstain |

Metrics: Hit@5, section recall@5, MRR, Precision@5, keyword answer score, abstention accuracy, latency and LLM calls per question.
The final cell compares query rewriting strategies (`none`, `rewrite`, `multi_query`, `hyde`, `adaptive`) on time vs accuracy.
It saves `data/eval_results/metrics_summary.md` with a table and resume bullets taken from your run.

## Chat app (FastAPI + SQL chat history)
```bash
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY
uvicorn app.main:app --reload
```
Then open http://127.0.0.1:8000. Chats and messages are stored in SQLite (`data/chat_history.db`, tables `chats` and `messages`).
To store them somewhere else, set `CHAT_DB_PATH`. To index a different folder of PDFs, set `RAG_DATA_DIR`.

API: `POST /api/chat {question, chat_id?, mode}`, `GET /api/chats`, `GET /api/chats/{id}/messages`, `DELETE /api/chats/{id}`.
