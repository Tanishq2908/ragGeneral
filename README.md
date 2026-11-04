# ragGeneral

Hybrid RAG over the PDFs in `data/`: MiniLM dense search + BM25, reciprocal rank fusion, cross-encoder reranking, and a Groq LLM to write the answer.

## Notebook: `notebook/doc.ipynb`
The last cells evaluate the pipeline on ~200 labelled questions. The set is chosen from the PDFs you loaded:

| file | corpus | labels | types |
|---|---|---|---|
| `data/eval_questions_legal.json` (200) | the legal PDFs in `data/` (Patents Act and Rules, TRIPS, CBD, Nagoya, WIPO GRATK, BD Act and Rules, Drugs & Cosmetics, DMR Act, FSSAI) | source file + exact evidence quote, each one checked against the PDF text | 159 single, 11 multi-document, 15 conversational, 15 unanswerable |
| `data/eval_questions.json` (201) | `AIcorpus.pdf` | section IDs A01–A48 | 133 single, 23 multi, 25 conversational, 4 distractor, 16 unanswerable |

Metrics: Hit@5, evidence/section recall@5, MRR, Precision@5, keyword answer score, abstention accuracy, latency and LLM calls per question.
The final cell compares query rewriting strategies (`none`, `rewrite`, `multi_query`, `hyde`, `adaptive`) on time vs accuracy.
It saves `data/eval_results/metrics_summary.md` with a table and resume bullets taken from your run.

## Chat app (FastAPI + SQL chat history)
```bash
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY
uvicorn app.main:app --reload
```
Then open http://127.0.0.1:8000 (the page lives in `frontend/index.html`). Chats and messages are stored in SQLite (`data/chat_history.db`, tables `chats` and `messages`).
To store them somewhere else, set `CHAT_DB_PATH`. To index a different folder of PDFs, set `RAG_DATA_DIR`.

API: `POST /api/chat {question, chat_id?, mode}`, `GET /api/chats`, `GET /api/chats/{id}/messages`, `DELETE /api/chats/{id}`.

## Deploy: backend on Render, frontend on Vercel
Vercel's serverless functions can't hold torch and the models (250 MB limit), and their disk doesn't keep a database.
So Render runs the API and Postgres, and Vercel serves the page and proxies `/api/*` to Render.

0. **Commit your PDFs into `data/`** (for example `data/AIcorpus.pdf`). The API indexes `data/**/*.pdf` at startup and won't start without them.
1. **Render**: Dashboard → New → Blueprint → choose this repo. `render.yaml` creates the `raggeneral-api` web service and the `raggeneral-db` Postgres database. Paste your `GROQ_API_KEY` when asked.
   Wait until `https://<your-service>.onrender.com/api/health` returns `{"ok": true, ...}`.
2. **Vercel**: New Project → import this repo → set **Root Directory = `frontend`** (Framework preset: Other, no build command) → Deploy.
   If your Render URL isn't `https://raggeneral-api.onrender.com`, change it in `frontend/vercel.json`.

The Render URL also serves the same UI by itself, so the Vercel step is optional.

Notes:
- The free Render plan has 512 MB RAM, so `USE_RERANKER=false` is set. On a plan with 1 GB or more, set it to `true` for better accuracy.
- Free Render services sleep when idle, so the first request after a sleep can take about a minute.
- Free Render Postgres databases expire after 30 days.
