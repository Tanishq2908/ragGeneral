# ragGeneral

Hybrid RAG over the PDFs in `data/`: MiniLM dense search + BM25, reciprocal rank fusion, cross-encoder reranking, and a Groq LLM to write the answer.

## Notebook: `notebook/doc.ipynb`
The last cells evaluate the pipeline on labelled questions. The set is chosen from the PDFs you loaded:

| file | corpus | labels | types |
|---|---|---|---|
| `data/eval_questions_legal.json` (114) | the Indian legal PDFs in `data/` (Patents Act and Rules, BD Act and Rules, Drugs & Cosmetics, DMR Act, FSSAI) | source file + exact evidence quote, each one checked against the PDF text | 88 single, 4 multi-document, 10 conversational, 12 unanswerable |
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

API: `POST /api/chat {question, chat_id?, mode}`, `GET /api/library` (indexed PDFs + modes), `GET /api/suggestions` (starter questions the LLM writes from your PDFs, cached), `GET /api/chats`, `GET /api/chats/{id}/messages`, `DELETE /api/chats/{id}`.

## Streamlit app (recommended, free on Streamlit Community Cloud, no card)
`ui/streamlit_app.py` is a Streamlit version of the chat ("Grove"): green theme with an animated tree, streaming answers,
chat history, starter questions generated from your PDFs, and NLP scores for every answer:

- **Per source passage:** semantic similarity (MiniLM cosine), BM25, TF-IDF cosine, keyword Jaccard overlap, query-term coverage,
  RRF fusion score, cross-encoder rerank score, rank among all passages, and the matched words. A sortable table compares them.
- **Per answer:** groundedness (answer vs sources, embedding cosine), answer relevance (answer vs question), lexical support
  (share of answer words found in the sources), TF-IDF answer-context overlap, valid citations,
  lexical diversity and top TF-IDF keywords.

Run locally:
```bash
pip install -r ui/requirements.txt
streamlit run ui/streamlit_app.py
```

Deploy:
1. Sign in at share.streamlit.io with GitHub → **Create app** → **Deploy a public app from GitHub**.
2. Repository `Tanishq2908/ragGeneral`, branch `main`, main file path `ui/streamlit_app.py`.
3. **Advanced settings**: Python 3.11, and in **Secrets** paste `GROQ_API_KEY = "your-key"`
   (optionally `DATABASE_URL = "postgresql://..."` from a free neon.tech database to keep chat history across restarts).
4. **Deploy**. The first start installs torch and indexes the PDFs, so give it several minutes.

## Deploy free on Hugging Face Spaces (no credit card)
A free Hugging Face Docker Space (2 vCPU, 16 GB RAM) runs the same `Dockerfile`. The GitHub Action
`.github/workflows/deploy-hf-space.yml` uploads the app to the Space on every push to `main`.

1. Sign up at huggingface.co (free, no card). Settings → Access Tokens → **Create new token** → type **Write** → copy it.
2. In this GitHub repo: Settings → Secrets and variables → Actions:
   - **Secrets** tab → New repository secret: `HF_TOKEN` = the Hugging Face token; `GROQ_API_KEY` = your Groq key.
   - **Variables** tab → New repository variable: `HF_SPACE` = `your-hf-username/raggeneral`.
3. Actions tab → **Deploy to Hugging Face Space** → **Run workflow** (or just push to `main`).
4. Open `https://huggingface.co/spaces/your-hf-username/raggeneral`; the first build takes ~10-15 min. The app itself is at `https://your-hf-username-raggeneral.hf.space`.

Notes: free Spaces sleep after 48 h without visitors (the next visit wakes them in a minute or two) and their disk is reset on restart,
so chat history is lost then. To keep it, create a free Postgres database at neon.tech (no card) and add its connection string
as a Space secret named `DATABASE_URL` (Space → Settings → Variables and secrets).

## Deploy on Google Cloud Run (recommended)
Cloud Run runs the `Dockerfile`: the image already contains the models and the PDF index (built at build time), so it starts in seconds.
Everything below runs in **Cloud Shell** (the `>_` button at the top of console.cloud.google.com), so you don't need to install anything.

**1. Get the code and pick a project/region** (asia-south1 = Mumbai):
```bash
git clone https://github.com/Tanishq2908/ragGeneral.git && cd ragGeneral
```
```bash
gcloud config set project YOUR_PROJECT_ID
```
```bash
export REGION=asia-south1
```

**2. Turn on the services and give the build enough time** (installing torch + indexing the PDFs takes ~10-15 min):
```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com sqladmin.googleapis.com
```
```bash
gcloud config set builds/timeout 3600
```

**3. Store your Groq key as a secret** (replace `YOUR_GROQ_KEY`; `printf` avoids a trailing newline that would break the key):
```bash
printf '%s' 'YOUR_GROQ_KEY' | gcloud secrets create groq-api-key --data-file=-
```
```bash
gcloud projects add-iam-policy-binding $(gcloud config get-value project) --member="serviceAccount:$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')-compute@developer.gserviceaccount.com" --role=roles/secretmanager.secretAccessor
```

**4. Deploy:**
```bash
gcloud run deploy raggeneral --source . --region $REGION --allow-unauthenticated --memory 2Gi --cpu 2 --cpu-boost --timeout 300 --set-env-vars USE_RERANKER=true --set-secrets GROQ_API_KEY=groq-api-key:latest
```
It prints a `https://raggeneral-....run.app` URL: that serves the chat UI and the API.

**5. (Optional but recommended) Keep chat history in Cloud SQL.** Without this, chats are stored in SQLite inside the container and disappear when Cloud Run restarts it. Use a password with only letters and numbers, since it goes into a URL.
```bash
gcloud sql instances create raggeneral-db --database-version=POSTGRES_16 --edition=ENTERPRISE --tier=db-f1-micro --region=$REGION --root-password=CHOOSE_A_PASSWORD
```
```bash
gcloud sql databases create chats --instance=raggeneral-db
```
```bash
printf 'postgresql://postgres:CHOOSE_A_PASSWORD@/chats?host=/cloudsql/%s' "$(gcloud sql instances describe raggeneral-db --format='value(connectionName)')" | gcloud secrets create database-url --data-file=-
```
```bash
gcloud run services update raggeneral --region $REGION --add-cloudsql-instances $(gcloud sql instances describe raggeneral-db --format='value(connectionName)') --update-secrets DATABASE_URL=database-url:latest
```

**Updating:** after changing code or PDFs, run step 4 again from an up-to-date clone (`git pull`).
**Costs:** Cloud Run scales to zero when idle (add `--min-instances 1` to avoid the ~10 s cold start, at extra cost). The `db-f1-micro` Cloud SQL instance costs roughly $8-10/month.
**Vercel frontend (optional):** set the Cloud Run URL in `frontend/vercel.json` and deploy the `frontend` folder as described below.

## Alternative: backend on Render, frontend on Vercel
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
