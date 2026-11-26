"""Grove: Streamlit chat UI for the RAG pipeline in app/rag_engine.py.

Run locally from the repo root:   streamlit run ui/streamlit_app.py
Deploy: Streamlit Community Cloud, main file path = ui/streamlit_app.py, secret GROQ_API_KEY (see README).
"""
import html
import os
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Streamlit secrets -> environment, before app.db / app.rag_engine read them at import time.
for key in ("GROQ_API_KEY", "DATABASE_URL", "USE_RERANKER", "CHAT_DB_PATH"):
    try:
        if not os.getenv(key) and key in st.secrets:
            os.environ[key] = str(st.secrets[key])
    except Exception:          # no secrets.toml when running locally
        pass

from app import db               # noqa: E402
from scenery import scenery_html  # noqa: E402

st.set_page_config(page_title="Grove · Chat with your PDFs", page_icon="🌳", layout="wide",
                   initial_sidebar_state="auto")

MODE_HELP = {
    "none": ("Fast", "Searches with your question as written. Quickest."),
    "rewrite": ("Rewrite", "The LLM rewrites the question into a cleaner search query first."),
    "multi_query": ("Multi-query", "Searches with 3 rephrasings of the question. Better recall, slower."),
    "hyde": ("HyDE", "Writes a draft answer and searches with it. Good for vague questions."),
}
LOGO = ('<svg viewBox="0 0 32 32" width="{s}" height="{s}" aria-hidden="true"><circle cx="16" cy="16" r="16" fill="#e1f0e2"/>'
        '<path d="M16 26V14" stroke="#4a3b2c" stroke-width="2" stroke-linecap="round"/>'
        '<path d="M16 15c-5 0-8-3-8-8 5 0 8 3 8 8Zm0 2c4 0 7-2.5 7-7-4 0-7 2.5-7 7Z" fill="#2f7d4f"/></svg>')

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Inter:wght@400;500;600&display=swap');
:root {
  --ink: #17261c; --ink-soft: #4f6355; --ink-faint: #7d8f81;
  --brand: #2f7d4f; --brand-strong: #1f5f3f; --brand-soft: #e1f0e2;
  --glass: rgba(255,255,255,.62); --glass-strong: rgba(255,255,255,.85); --line: rgba(31,95,63,.14);
  --leaf-1: #2f7d4f; --leaf-2: #4f9f5f; --leaf-3: #7fbf6a; --bark: #4a3b2c;
  --shadow: 0 10px 30px -12px rgba(23,60,38,.25);
}
html, body, .stApp, button, input, textarea { font-family: Inter, system-ui, sans-serif; }
.stApp {
  background: radial-gradient(1200px 700px at 85% 110%, #dcebd8, transparent 60%),
              radial-gradient(900px 600px at 0% -10%, #dcebd8, transparent 55%), #eef5ec;
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stMain"], [data-testid="stSidebar"], [data-testid="stBottom"], header[data-testid="stHeader"] { position: relative; z-index: 1; }
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent !important; }
.block-container { max-width: 1440px; padding-top: 2.2rem; padding-bottom: 7rem; }
.st-key-sugg_wrap { max-width: 820px; margin: 0 auto; }
[data-testid="stBottomBlockContainer"] { max-width: 1440px; }

/* sidebar */
[data-testid="stSidebar"] { background: var(--glass); backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); border-right: 1px solid var(--line); }
.grove-brand { display: flex; align-items: center; gap: 10px; margin: -6px 0 6px; }
.grove-brand b { font: 600 23px/1 Fraunces, Georgia, serif; color: var(--brand-strong); letter-spacing: -.01em; }
.grove-brand small { display: block; font-size: 12px; color: var(--ink-faint); margin-top: 4px; }
.grove-label { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-faint); margin: 14px 0 4px; }
[data-testid="stSidebar"] .stButton button { border-radius: 12px; }
[data-testid="stSidebar"] [class*="st-key-chat_"] button {
  justify-content: flex-start; text-align: left; border: 0; background: transparent; color: var(--ink-soft);
  padding: 6px 10px; min-height: 0; box-shadow: none;
}
[data-testid="stSidebar"] [class*="st-key-chat_"] button p { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
[data-testid="stSidebar"] [class*="st-key-chat_"] button:hover { background: var(--brand-soft); color: var(--ink); }
[data-testid="stSidebar"] [class*="st-key-chat_"] button[kind="primary"] { background: var(--brand-soft); color: var(--brand-strong); font-weight: 600; box-shadow: inset 3px 0 0 var(--brand); }
[data-testid="stSidebar"] [class*="st-key-del_"] button { border: 0; background: transparent; color: var(--ink-faint); padding: 6px 4px; min-height: 0; }
[data-testid="stSidebar"] [class*="st-key-del_"] button:hover { color: #c0392b; background: rgba(192,57,43,.08); }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { gap: 2px; align-items: center; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width: 0 !important; width: auto !important; flex: 1 1 0 !important; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child { flex: 0 0 34px !important; }
.grove-doc { display: flex; justify-content: space-between; gap: 8px; font-size: 13px; color: var(--ink-soft); padding: 3px 0; }
.grove-doc em { font-style: normal; color: var(--ink-faint); white-space: nowrap; }

/* hero */
.grove-hero svg { display: block; margin: 0 auto; }
.grove-hero { text-align: center; margin: 7vh auto 22px; max-width: 620px; animation: grove-rise .5s ease both; }
.grove-hero h1 { font: 600 clamp(28px, 4vw, 38px)/1.15 Fraunces, Georgia, serif !important; color: var(--brand-strong) !important; margin: 14px 0 8px; padding: 0; letter-spacing: -.015em; }
.grove-hero p { color: var(--ink-soft); margin: 0; }
[class*="st-key-sugg_"] button {
  width: 100%; min-height: 64px; text-align: left; justify-content: flex-start; border-radius: 14px; border: 1px solid var(--line);
  background: var(--glass); backdrop-filter: blur(10px); color: var(--ink-soft); transition: transform .15s ease, border-color .15s;
}
[class*="st-key-sugg_"] button:hover { transform: translateY(-2px); border-color: var(--brand); color: var(--ink); background: var(--glass-strong); }
.grove-skeleton { height: 64px; border-radius: 14px; border: 1px solid var(--line); margin-bottom: 12px;
  background: linear-gradient(90deg, var(--glass) 0%, var(--brand-soft) 50%, var(--glass) 100%); background-size: 200% 100%; animation: grove-shimmer 1.4s infinite; }
@keyframes grove-shimmer { from { background-position: 100% 0; } to { background-position: -100% 0; } }
@keyframes grove-rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

/* messages */
[data-testid="stChatMessage"] {
  background: var(--glass-strong); border: 1px solid var(--line); border-radius: 18px; padding: 14px 16px;
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); box-shadow: var(--shadow); margin-bottom: 14px; animation: grove-rise .35s ease both;
}
[class*="st-key-umsg"] [data-testid="stChatMessage"] {
  flex-direction: row-reverse; margin-left: auto; gap: 10px; width: fit-content; max-width: 88%;
  background: linear-gradient(135deg, #2f7d4f, #3f9963); border: 0; color: #fff;
}
[class*="st-key-umsg"] [data-testid="stChatMessage"] * { color: #fff !important; }
[class*="st-key-umsg"] [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] { text-align: left; }
[class*="st-key-umsg"] [data-testid="stChatMessage"] > div:first-child { background: rgba(255,255,255,.22); border: 0; }
[class*="st-key-amsg"] [data-testid="stChatMessage"] > div:first-child { background: var(--brand-soft); border: 0; }
.grove-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.grove-chip { font-size: 11.5px; padding: 2px 9px; border-radius: 999px; background: var(--brand-soft); color: var(--brand-strong); }
.grove-src { margin: 8px 0 0; padding: 9px 12px; border-radius: 10px; border-left: 3px solid var(--brand); background: var(--brand-soft); font-size: 13px; color: var(--ink-soft); }
.grove-notfound { padding: 12px 14px; border-radius: 12px; background: #fff7e6; border: 1px solid #f3dca6; color: #6b4e00; font-size: 14.5px; line-height: 1.5; }
.grove-notfound b { color: #5a3f00; }
.grove-src { margin-bottom: 10px; }
.grove-src .grove-meta { margin: 6px 0 4px; }
.grove-chip b { font-weight: 600; }
.grove-chip.strong, .grove-chip.strong b { background: #2f7d4f; color: #fff; }
.grove-terms { font-size: 12px; color: var(--ink-faint); margin: 4px 0; }
.grove-term { display: inline-block; font-size: 11.5px; padding: 1px 8px; margin: 2px 3px 0 0; border-radius: 6px; background: #fff7d6; color: #6b5300; border: 1px solid #f0e2a8; }
.grove-preview { margin-top: 4px; }
.grove-tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; }
.grove-tile { padding: 10px 12px; border-radius: 12px; background: var(--brand-soft); border: 1px solid var(--line); }
.grove-tile-label { font-size: 11px; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; color: var(--ink-faint); }
.grove-tile-value { font: 600 22px/1.2 Fraunces, Georgia, serif; color: var(--brand-strong); margin: 2px 0 6px; }
.grove-bar { height: 6px; border-radius: 99px; background: rgba(31,95,63,.12); overflow: hidden; }
.grove-bar i { display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, #7fbf6a, #2f7d4f); }
.grove-tile-note { font-size: 11.5px; color: var(--ink-faint); margin-top: 5px; }
.grove-src b { color: var(--ink); } .grove-src em { font-style: normal; color: var(--ink-faint); font-size: 12px; }
[data-testid="stChatMessage"] [data-testid="stExpander"] details { border: 0; background: transparent; }
[data-testid="stChatMessage"] [data-testid="stExpander"] summary { padding-left: 0; color: var(--brand-strong); font-weight: 600; font-size: 13px; }

/* input */
[data-testid="stChatInput"] { border-radius: 20px !important; border: 1px solid var(--line) !important; background: var(--glass-strong) !important;
  backdrop-filter: blur(18px); box-shadow: var(--shadow); }
[data-testid="stChatInput"]:focus-within { border-color: var(--brand) !important; box-shadow: var(--shadow), 0 0 0 4px var(--brand-soft); }
[data-testid="stChatInputSubmitButton"] { background: linear-gradient(135deg, #2f7d4f, #3f9963) !important; color: #fff !important; border-radius: 12px !important; }

/* answer formatting */
[class*="st-key-amsg"] [data-testid="stChatMessageContent"] { line-height: 1.6; }
[class*="st-key-amsg"] [data-testid="stChatMessageContent"] h3 { font: 600 1.02rem/1.3 Inter, sans-serif; color: var(--brand-strong); margin: .9rem 0 .3rem; padding: 0; }
[class*="st-key-amsg"] [data-testid="stChatMessageContent"] ul, [class*="st-key-amsg"] [data-testid="stChatMessageContent"] ol { margin: .2rem 0 .5rem; padding-left: 1.2rem; }
[class*="st-key-amsg"] [data-testid="stChatMessageContent"] li { margin: .15rem 0; }
[class*="st-key-amsg"] [data-testid="stChatMessageContent"] strong { color: var(--ink); }
[class*="st-key-amsgf"] [data-testid="stChatMessage"] { border-color: rgba(47,125,79,.55); box-shadow: 0 0 0 3px var(--brand-soft), var(--shadow); }
[class*="st-key-focus_"] button { border: 0; background: transparent; color: var(--brand-strong); padding: 2px 8px; min-height: 0; font-size: 12.5px; box-shadow: none; }
[class*="st-key-focus_"] button:hover { background: var(--brand-soft); color: var(--brand-strong); }
[class*="st-key-amsgf"] [class*="st-key-focus_"] button { background: var(--brand-soft); font-weight: 600; }
[class*="st-key-inline_"] { display: none; }

/* insights panel */
[data-testid="stHorizontalBlock"]:has(.st-key-insights_panel) { align-items: flex-start; }
[data-testid="stColumn"]:has(.st-key-insights_panel) { position: sticky; top: 3.75rem; align-self: flex-start; z-index: 2; }
.st-key-insights_panel {
  max-height: calc(100vh - 10rem); overflow-y: auto; overscroll-behavior: contain;
  background: var(--glass-strong); border: 1px solid var(--line); border-radius: 18px; padding: 16px 16px 8px;
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); box-shadow: var(--shadow);
}
.grove-panel-head { margin-bottom: 2px; }
.grove-panel-head .grove-label { margin-top: 0; }
.grove-panel-q { font: 600 15px/1.4 Inter, sans-serif; color: var(--ink); }
.grove-sub { font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-faint); margin: 14px 0 6px; }
.grove-note { font-size: 12.5px; color: var(--ink-faint); margin: 8px 0; line-height: 1.45; }
.grove-steps { margin: 0 0 4px; padding-left: 1.2rem; font-size: 13px; color: var(--ink-soft); }
.grove-steps li { margin: 3px 0; }
.grove-kv { display: flex; justify-content: space-between; gap: 10px; font-size: 13px; color: var(--ink-soft); padding: 5px 0; border-bottom: 1px dashed var(--line); }
.grove-kv b { color: var(--ink); font-weight: 600; }
.grove-src-head { display: flex; align-items: baseline; gap: 8px; }
.grove-src-head b { flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.grove-num { display: inline-grid; place-items: center; min-width: 20px; height: 20px; border-radius: 6px; background: #2f7d4f; color: #fff; font-size: 11px; font-weight: 700; }
.grove-bar.thin { height: 4px; margin: 7px 0 2px; }
.st-key-insights_panel .grove-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.st-key-insights_panel .grove-preview { font-size: 12.5px; color: var(--ink-soft); }

@media (max-width: 1000px) {
  [data-testid="stColumn"]:has(> div > div > .st-key-insights_panel),
  [data-testid="stColumn"]:has(.st-key-insights_panel) { display: none !important; }
  [data-testid="stHorizontalBlock"]:has(.st-key-insights_panel) > [data-testid="stColumn"]:first-child { width: 100% !important; flex: 1 1 100% !important; }
  [class*="st-key-inline_"] { display: block; }
  [class*="st-key-focus_"] { display: none; }
}

@media (max-width: 760px) {
  .block-container { padding-left: 12px; padding-right: 12px; padding-top: 3rem; }
  .grove-hero { margin-top: 2vh; }
  [class*="st-key-umsg"] [data-testid="stChatMessage"] { max-width: 94%; }
}
</style>
"""


# ---------- resources ----------
@st.cache_resource(show_spinner=False)
def get_engine():
    from app.rag_engine import RagEngine
    return RagEngine()


@st.cache_resource(show_spinner=False)
def ensure_db():
    db.init_db()
    return True


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def get_suggestions():
    return get_engine().suggestions()


def esc(s):
    return html.escape(str(s))


def doc_label(name):
    return Path(name).stem.replace("_", " ").replace("-", " ")


def mode_label(mode):
    return MODE_HELP.get(mode or "none", (mode or "Fast",))[0]


SCORE_INFO = {   # key: (label, explanation, format)
    "dense_cosine": ("Semantic", "Cosine similarity of MiniLM embeddings (question vs passage), -1 to 1. Meaning, not exact words.", "{:.2f}"),
    "bm25": ("BM25", "Okapi BM25 keyword relevance: rare question words that appear often in the passage score high. Unbounded.", "{:.2f}"),
    "tfidf_cosine": ("TF-IDF", "Cosine similarity of TF-IDF vectors (term frequency x inverse document frequency), 0 to 1.", "{:.2f}"),
    "keyword_jaccard": ("Jaccard", "Jaccard overlap of content words: shared words / all words in both, 0 to 1.", "{:.0%}"),
    "query_term_coverage": ("Coverage", "Share of the question's content words that appear in the passage.", "{:.0%}"),
    "rrf_fusion": ("RRF", "Reciprocal Rank Fusion score combining the semantic and BM25 rankings (sum of 1/(60+rank)).", "{:.4f}"),
    "rerank": ("Rerank", "Cross-encoder relevance score (ms-marco MiniLM) for the question-passage pair. Higher is better.", "{:.2f}"),
}
KEY_SCORES = ("dense_cosine", "bm25", "tfidf_cosine")     # shown on each source card; the rest are in the table


def chip(label, value=None, tip="", strong=False):
    inner = esc(label) + (f" <b>{esc(value)}</b>" if value is not None else "")
    return f'<span class="grove-chip{" strong" if strong else ""}" title="{esc(tip)}">{inner}</span>'


def footer_html(m):
    plan = (m.get("analysis") or {}).get("plan") or {}
    chips = [chip(mode_label(m.get("mode")))]
    if m.get("latency_ms") is not None:
        chips.append(chip(f"{m['latency_ms'] / 1000:.1f}s"))
    n = len(m.get("sources") or [])
    if n:
        chips.append(chip(f"{n} source{'s' if n != 1 else ''}"))
    if plan.get("sub_questions"):
        chips.append(chip(f"{len(plan['sub_questions'])}-part question", tip="The question was split into parts, "
                          "and each part was searched separately."))
    return '<div class="grove-meta">' + "".join(chips) + "</div>"


def show_not_found():
    st.html('<div class="grove-notfound"><b>No answer found in your documents.</b><br>'
            'None of the passages that were searched answer this. Try naming the Act or Rules, using different '
            'words, or asking something more specific.</div>')


def source_cards_html(sources):
    cards = []
    for i, s in enumerate(sources, 1):
        sc = s.get("scores") or {}
        rel = max(0.0, min(1.0, sc.get("dense_cosine", 0.0)))
        chips_ = "".join(chip(SCORE_INFO[k][0], SCORE_INFO[k][2].format(sc[k]), SCORE_INFO[k][1], strong=k == "dense_cosine")
                         for k in KEY_SCORES if k in sc)
        terms = "".join(f'<span class="grove-term">{esc(t)}</span>' for t in sc.get("matched_terms", [])[:6])
        preview = s.get("preview", "")
        cards.append(
            f'<div class="grove-src"><div class="grove-src-head"><span class="grove-num">{i}</span>'
            f'<b title="{esc(s["source"])}">{esc(doc_label(s["source"]))}</b><em>p.{s["page"] + 1}</em></div>'
            f'<div class="grove-bar thin" title="Semantic similarity {rel:.2f}"><i style="width:{rel * 100:.0f}%"></i></div>'
            f'<div class="grove-meta">{chips_}</div>'
            f'{"<div class=grove-terms>" + terms + "</div>" if terms else ""}'
            f'<div class="grove-preview">{esc(preview[:220])}{"…" if len(preview) > 0 else ""}</div></div>')
    return "".join(cards)


def scores_table(sources):
    rows = []
    for i, s in enumerate(sources, 1):
        sc = s.get("scores") or {}
        row = {"#": i, "Document": doc_label(s["source"]), "Page": s["page"] + 1}
        row.update({SCORE_INFO[k][0]: sc[k] for k in SCORE_INFO if k in sc})
        if "dense_rank" in sc:
            row.update({"Semantic rank": sc["dense_rank"], "BM25 rank": sc["bm25_rank"]})
        rows.append(row)
    unit = {"Semantic": (-1.0, 1.0), "TF-IDF": (0.0, 1.0), "Jaccard": (0.0, 1.0), "Coverage": (0.0, 1.0)}
    cfg = {label: st.column_config.ProgressColumn(label, min_value=lo, max_value=hi, format="%.2f",
                                                  help=next(v[1] for v in SCORE_INFO.values() if v[0] == label))
           for label, (lo, hi) in unit.items()}
    cfg.update({v[0]: st.column_config.NumberColumn(v[0], help=v[1], format="%.3f")
                for k, v in SCORE_INFO.items() if v[0] not in unit})
    st.dataframe(rows, hide_index=True, use_container_width=True, column_config=cfg)
    st.caption("Hover a column name to see what it means. Click a column to sort.")


def meter(label, value, tip, lo=0.0, hi=1.0, fmt="{:.2f}", note=None):
    pct = max(0.0, min(1.0, (value - lo) / (hi - lo))) * 100 if value is not None else 0
    shown = fmt.format(value) if value is not None else "–"
    return (f'<div class="grove-tile" title="{esc(tip)}"><div class="grove-tile-label">{esc(label)}</div>'
            f'<div class="grove-tile-value">{esc(shown)}</div>'
            f'<div class="grove-bar"><i style="width:{pct:.0f}%"></i></div>'
            f'<div class="grove-tile-note">{esc(note or "")}</div></div>')


def quality_html(a, n_sources):
    cites, valid = a.get("citations", []), a.get("valid_citations", [])
    tiles = [
        meter("Groundedness", a.get("groundedness_max"), "Highest embedding cosine between the answer and a retrieved "
              "passage. High = the answer stays close to the source text.", note="answer vs best source"),
        meter("Relevance", a.get("answer_relevance"), "Embedding cosine between the answer and the question.",
              note="answer vs question"),
        meter("Lexical support", a.get("lexical_support"), "Share of the answer's content words that also appear in "
              "the sources.", fmt="{:.0%}", note="answer words in sources"),
        meter("Citations", len(valid), "[Source n] references in the answer that point to a retrieved passage.",
              hi=max(1, n_sources), fmt="{:d}", note=f"{len(valid)} valid of {len(cites)}"),
    ]
    more = [("TF-IDF overlap (answer vs sources)", a.get("tfidf_answer_context"), "{:.2f}"),
            ("Semantic mean (answer vs all sources)", a.get("groundedness_mean"), "{:.2f}"),
            ("Lexical diversity (type-token ratio)", a.get("lexical_diversity"), "{:.2f}"),
            ("Words / sentences", f'{a.get("word_count", 0)} / {a.get("sentence_count", 0)}', "{}")]
    rows = "".join(f'<div class="grove-kv"><span>{esc(k)}</span><b>{esc(f.format(v) if v is not None else "–")}</b></div>'
                   for k, v, f in more)
    kw = "".join(f'<span class="grove-term">{esc(k)}</span>' for k in a.get("top_keywords", []))
    return (f'<div class="grove-tiles">{"".join(tiles)}</div>'
            f'<div class="grove-sub">More metrics</div>{rows}'
            + (f'<div class="grove-sub">Top TF-IDF keywords</div><div class="grove-terms">{kw}</div>' if kw else ""))


def plan_html(m, plan):
    parts = plan.get("sub_questions") or []
    queries = [q for q in plan.get("search_queries") or [] if q]
    out = ['<div class="grove-meta">' + chip("Search", mode_label(m.get("mode")))
           + (chip("Time", f'{m["latency_ms"] / 1000:.1f}s') if m.get("latency_ms") is not None else "") + "</div>"]
    if parts:
        out.append('<div class="grove-sub">Question split into parts</div><ol class="grove-steps">'
                   + "".join(f"<li>{esc(p)}</li>" for p in parts) + "</ol>")
    else:
        out.append('<div class="grove-note">Searched as a single question.</div>')
    if len(queries) > 1 or (queries and queries[0] != m.get("question")):
        out.append('<div class="grove-sub">Search queries</div><ul class="grove-steps">'
                   + "".join(f"<li>{esc(q)}</li>" for q in queries[:6]) + "</ul>")
    out.append('<div class="grove-note">Each part is searched with BM25 and MiniLM embeddings, fused with Reciprocal '
               'Rank Fusion and re-ranked; when a part names an Act or Rules, its search stays inside that document.</div>')
    return "".join(out)


def insights_body(m):
    """Tabs with the sources, answer quality and search plan of one assistant message."""
    a = m.get("analysis") or {}
    sources = m.get("sources") or []
    not_found = is_abstain(m["content"])
    t_src, t_quality, t_plan = st.tabs([f"Sources ({len(sources)})", "Quality", "Plan"])
    with t_src:
        if not_found:
            st.caption("Closest passages searched. None of them answered the question.")
        if sources:
            st.html(source_cards_html(sources))
            with st.expander("Compare all scores"):
                scores_table(sources)
        else:
            st.caption("No passages.")
    with t_quality:
        if not_found or not a or "groundedness_max" not in a:
            st.caption("No quality scores: the documents did not contain an answer.")
        else:
            st.html(quality_html(a, len(sources)))
    with t_plan:
        st.html(plan_html(m, a.get("plan") or {}))


def show_message(i, m, focused=False):
    role = m["role"]
    box = st.container(key=f"{'umsg' if role == 'user' else ('amsgf' if focused else 'amsg')}_{i}")
    with box, st.chat_message(role, avatar="🧑" if role == "user" else "🌳"):
        if role == "user":
            st.markdown(m["content"])
            return
        if is_abstain(m["content"]):
            show_not_found()
        else:
            st.markdown(m["content"])
        st.html(footer_html(m))
        if st.button("📊 Sources & scores", key=f"focus_{i}", help="Show this answer's sources and scores in the side panel"):
            state.focus = i
            st.rerun()
        with st.container(key=f"inline_{i}"):            # phones: the side panel is hidden, details fold out here
            with st.expander("Sources & scores"):
                insights_body(m)


def panel_header(question_text):
    st.html(f'<div class="grove-panel-head"><div class="grove-label">Answer insights</div>'
            f'<div class="grove-panel-q">{esc(question_text[:160])}{"…" if len(question_text) > 160 else ""}</div></div>')


def chat_markdown(msgs):
    lines = ["# Grove chat export", ""]
    for m in msgs:
        lines += [f"**{'You' if m['role'] == 'user' else 'Grove'}:**", "", m["content"], ""]
        if m["role"] == "assistant" and m.get("sources"):
            lines += ["Sources: " + "; ".join(f"[{i}] {s['source']} p.{s['page'] + 1}"
                                              for i, s in enumerate(m["sources"], 1)), ""]
    return "\n".join(lines)


# ---------- page ----------
st.html(STYLE)
_scenery_css, _scenery_markup = scenery_html()
st.html(_scenery_css)
st.markdown(_scenery_markup, unsafe_allow_html=True)
ensure_db()

state = st.session_state
state.setdefault("chat_id", None)
state.setdefault("mode", "none")
state.setdefault("pending", None)
state.setdefault("focus", None)

if "engine_ready" not in state:
    with st.status("🌱 Getting the library ready…", expanded=True) as status:
        st.write("Loading the search models and the index of your PDFs. "
                 "The very first start after a deploy reads and embeds every PDF, which can take a few minutes.")
        try:
            get_engine()
        except Exception as e:
            status.update(label="Couldn't load the documents", state="error")
            st.error(f"{e}")
            st.stop()
        status.update(label="Library ready", state="complete", expanded=False)
    state.engine_ready = True
    st.rerun()

engine = get_engine()
from app.rag_engine import MODES, clean_answer, fallback_modes, is_abstain  # noqa: E402

messages = db.get_messages(state.chat_id) if state.chat_id and db.chat_exists(state.chat_id) else []
if state.chat_id and not messages and not db.chat_exists(state.chat_id):
    state.chat_id = None

# ---------- sidebar ----------
with st.sidebar:
    n_docs = len(engine.documents)
    st.markdown(f'<div class="grove-brand">{LOGO.format(s=34)}<div><b>Grove</b>'
                f'<small>{n_docs} document{"s" if n_docs != 1 else ""} indexed</small></div></div>', unsafe_allow_html=True)
    if st.button("＋  New chat", type="primary", use_container_width=True):
        state.chat_id, state.focus = None, None
        st.rerun()

    st.html('<div class="grove-label">Recent</div>')
    chats = db.list_chats()
    if not chats:
        st.caption("No chats yet. Ask your first question.")
    for c in chats[:40]:
        col_title, col_del = st.columns([0.86, 0.14])
        with col_title:
            if st.button(c["title"], key=f"chat_{c['id']}", use_container_width=True, help=c["title"],
                         type="primary" if c["id"] == state.chat_id else "secondary"):
                state.chat_id, state.focus = c["id"], None
                st.rerun()
        with col_del:
            if st.button("✕", key=f"del_{c['id']}", help="Delete chat"):
                db.delete_chat(c["id"])
                if state.chat_id == c["id"]:
                    state.chat_id, state.focus = None, None
                st.rerun()

    st.html('<div class="grove-label">Search mode</div>')
    state.mode = st.selectbox("Search mode", MODES, index=MODES.index(state.mode) if state.mode in MODES else 0,
                              format_func=mode_label, label_visibility="collapsed",
                              help="\n\n".join(f"**{v[0]}**: {v[1]}" for k, v in MODE_HELP.items() if k in MODES))
    st.caption(MODE_HELP.get(state.mode, ("", ""))[1] + " Multi-part questions are split automatically.")

    with st.expander(f"📚 Your library ({n_docs})"):
        st.html("".join(f'<div class="grove-doc" title="{esc(d["name"])}"><span>{esc(doc_label(d["name"]))}</span>'
                        f'<em>{d["pages"]} p</em></div>' for d in engine.documents))
    if messages:
        st.download_button("⬇  Export this chat (.md)", chat_markdown(messages), file_name="grove-chat.md",
                           mime="text/markdown", use_container_width=True)

# ---------- main ----------
question = st.chat_input("Ask anything about your documents…")
if state.pending and not question:
    question, state.pending = state.pending, None

if not messages and not question:
    st.markdown(f'<div class="grove-hero">{LOGO.format(s=64)}<h1>What would you like to know?</h1>'
                f'<p>Ask about the Acts, Rules and Regulations in your library. Answers cite the document and page; '
                f'multi-part questions are split and each part is searched.</p></div>', unsafe_allow_html=True)
    with st.container(key="sugg_wrap"):
        sugg_slot = st.empty()
        with sugg_slot.container():
            st.html('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px">'
                    + '<div class="grove-skeleton"></div>' * 4 + "</div>")
        try:
            suggestions = get_suggestions()
        except Exception:
            suggestions = []
        with sugg_slot.container():
            cols = st.columns(2)
            for i, q in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(q, key=f"sugg_{i}", use_container_width=True):
                        state.pending = q
                        st.rerun()
    st.stop()

# attach each answer's question (for the panel header and plan)
for k, m in enumerate(messages):
    if m["role"] == "assistant":
        m["question"] = next((p["content"] for p in reversed(messages[:k]) if p["role"] == "user"), "")
answers = [k for k, m in enumerate(messages) if m["role"] == "assistant"]
focus = state.focus if state.focus in answers else (answers[-1] if answers else None)

chat_col, panel_col = st.columns([0.6, 0.4], gap="large")
st.empty()          # takes the slot of the start screen's suggestions so they don't linger while answering
with panel_col:
    panel = st.container(key="insights_panel")

with chat_col:
    for k, m in enumerate(messages):
        show_message(k, m, focused=(k == focus and not question))

if not question:
    with panel:
        if focus is not None:
            panel_header(messages[focus]["question"])
            insights_body(messages[focus])
        else:
            st.html('<div class="grove-panel-head"><div class="grove-label">Answer insights</div>'
                    '<div class="grove-note">Sources and scores for an answer appear here.</div></div>')
    st.stop()

# ---------- new question ----------
question = question.strip()[:2000]
if state.chat_id is None:
    state.chat_id = db.create_chat(question)
db.add_message(state.chat_id, "user", question, mode=state.mode)
with chat_col:
    show_message(len(messages), {"role": "user", "content": question})
with panel:
    panel_header(question)
    live = st.empty()
    live.html('<div class="grove-note">🔎 Planning the search…</div>')

with chat_col, st.container(key="amsgf_live"), st.chat_message("assistant", avatar="🌳"):
    t0 = time.perf_counter()
    note = st.empty()
    used = {"mode": state.mode, "prep": None}

    def show_live(prep, mode, status):
        m = {"mode": mode, "question": question, "latency_ms": None}
        live.html(f'<div class="grove-note">{esc(status)}</div>'
                  + plan_html(m, {"sub_questions": prep["sub_questions"], "search_queries": prep["search_queries"]})
                  + f'<div class="grove-sub">Sources found ({len(prep["docs"])})</div>'
                  + source_cards_html(engine.sources(prep["docs"])))

    def answer_stream():
        """Streams the answer. If the chosen search finds nothing, silently retries with a broader one."""
        modes = fallback_modes(state.mode)
        for n, mode in enumerate(modes):
            note.caption("🔎 Searching your documents…" if not n else
                         f"🔁 Not found with {mode_label(modes[n - 1])} search. Trying {mode_label(mode)} search…")
            prep = engine.prepare(question, mode, messages)
            used.update(mode=mode, prep=prep)
            show_live(prep, mode, "✍️ Writing the answer from these sources…")
            pieces = engine.stream_answer(question, prep["docs"], prep["sub_questions"])
            head = ""
            for piece in pieces:                       # look ahead: is the model abstaining?
                head += piece
                if len(head) >= 24:
                    break
            if is_abstain(head):
                for _ in pieces:
                    pass
                if n < len(modes) - 1:
                    continue
                note.empty()
                return                                 # shown as a "not found" card below
            note.empty()
            yield head
            yield from pieces
            return

    try:
        streamed = st.write_stream(answer_stream())
        answer = clean_answer(streamed if isinstance(streamed, str) else "".join(map(str, streamed or [])))
    except Exception as e:
        answer = None
        msg = str(e)
        if "api_key" in msg.lower() or "401" in msg:
            msg = "The Groq API key is missing or invalid. Add GROQ_API_KEY in the app's secrets."
        elif "rate" in msg.lower() and "limit" in msg.lower():
            msg = "Groq's free rate limit was reached. Wait a minute and ask again."
        st.error(f"Something went wrong: {msg}")
    if answer is not None:
        latency = round((time.perf_counter() - t0) * 1000)
        prep, mode = used["prep"], used["mode"]
        docs = prep["docs"] if prep else []
        try:
            analysis = engine.analyze(question, answer, docs)
        except Exception:
            analysis = {}
        analysis["plan"] = {"sub_questions": prep["sub_questions"] if prep else [],
                            "search_queries": prep["search_queries"] if prep else []}
        if is_abstain(answer):
            show_not_found()
        db.add_message(state.chat_id, "assistant", answer, mode=mode, sources=engine.sources(docs),
                       latency_ms=latency, analysis=analysis)
        state.focus = None                              # focus the newest answer
        st.rerun()
