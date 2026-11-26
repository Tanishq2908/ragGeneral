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
.block-container { max-width: 860px; padding-top: 2.2rem; padding-bottom: 7rem; }

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


def meta_html(mode, latency_ms):
    chips = [MODE_HELP.get(mode or "none", (mode,))[0]]
    if latency_ms is not None:
        chips.append(f"{latency_ms / 1000:.2f}s")
    return '<div class="grove-meta">' + "".join(f'<span class="grove-chip">{esc(c)}</span>' for c in chips) + "</div>"


SCORE_INFO = {   # key: (label, explanation, format)
    "dense_cosine": ("Semantic", "Cosine similarity of MiniLM embeddings (question vs passage), -1 to 1. Meaning, not exact words.", "{:.2f}"),
    "bm25": ("BM25", "Okapi BM25 keyword relevance: rare question words that appear often in the passage score high. Unbounded.", "{:.2f}"),
    "tfidf_cosine": ("TF-IDF", "Cosine similarity of TF-IDF vectors (term frequency x inverse document frequency), 0 to 1.", "{:.2f}"),
    "keyword_jaccard": ("Jaccard", "Jaccard overlap of content words: shared words / all words in both, 0 to 1.", "{:.0%}"),
    "query_term_coverage": ("Coverage", "Share of the question's content words that appear in the passage.", "{:.0%}"),
    "rrf_fusion": ("RRF", "Reciprocal Rank Fusion score combining the semantic and BM25 rankings (sum of 1/(60+rank)).", "{:.4f}"),
    "rerank": ("Rerank", "Cross-encoder relevance score (ms-marco MiniLM) for the question-passage pair. Higher is better.", "{:.2f}"),
}


def chip(label, value, tip="", strong=False):
    return (f'<span class="grove-chip{" strong" if strong else ""}" title="{esc(tip)}">'
            f'{esc(label)} <b>{esc(value)}</b></span>')


def score_chips(sc):
    out = []
    for key, (label, tip, fmt) in SCORE_INFO.items():
        if key in sc:
            out.append(chip(label, fmt.format(sc[key]), tip, strong=key in ("dense_cosine", "rerank")))
    if "dense_rank" in sc:
        out.append(chip("Ranks", f'#{sc["dense_rank"]} sem · #{sc["bm25_rank"]} BM25',
                        "Where this passage ranks among all passages by semantic similarity and by BM25 alone."))
    return '<div class="grove-meta">' + "".join(out) + "</div>"


def show_not_found():
    st.html('<div class="grove-notfound"><b>No answer found in your documents.</b><br>'
            'None of the passages that were searched answer this. Try naming the Act or Rules, using different '
            'words, or asking something more specific.</div>')


def show_sources(sources, not_found=False):
    if not sources:
        return
    n = len(sources)
    title = (f"🔎 {'Closest passage' if n == 1 else f'{n} closest passages'} searched (none answered it)" if not_found
             else f"📄 {len(sources)} source{'s' if len(sources) > 1 else ''} · retrieval scores")
    with st.expander(title):
        tab_cards, tab_table = st.tabs(["Passages", "Compare scores"])
        with tab_cards:
            cards = []
            for i, s in enumerate(sources, 1):
                sc = s.get("scores") or {}
                terms = "".join(f'<span class="grove-term">{esc(t)}</span>' for t in sc.get("matched_terms", []))
                cards.append(
                    f'<div class="grove-src"><b>[{i}] {esc(s["source"])}</b> <em>· p.{s["page"] + 1}</em>'
                    f'{score_chips(sc) if sc else ""}'
                    f'{"<div class=grove-terms>Matched words: " + terms + "</div>" if terms else ""}'
                    f'<div class="grove-preview">{esc(s["preview"])}…</div></div>')
            st.html("".join(cards))
        with tab_table:
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
            st.caption("Hover a column name to see what the score means. Click a column to sort.")


def meter(label, value, tip, lo=0.0, hi=1.0, fmt="{:.2f}", good=None):
    pct = max(0.0, min(1.0, (value - lo) / (hi - lo))) * 100 if value is not None else 0
    shown = fmt.format(value) if value is not None else "–"
    return (f'<div class="grove-tile" title="{esc(tip)}"><div class="grove-tile-label">{esc(label)}</div>'
            f'<div class="grove-tile-value">{esc(shown)}</div>'
            f'<div class="grove-bar"><i style="width:{pct:.0f}%"></i></div>'
            f'<div class="grove-tile-note">{esc(good or "")}</div></div>')


def show_analysis(a, n_sources):
    if not a or a.get("abstained"):
        return
    with st.expander("📊 Answer analysis · NLP scores"):
        if a.get("abstained"):
            st.info("The model abstained: it found no answer in the retrieved passages.")
        cites, valid = a.get("citations", []), a.get("valid_citations", [])
        tiles = [
            meter("Groundedness", a.get("groundedness_max"), "Highest embedding cosine between the answer and a retrieved passage. "
                  "High = the answer stays close to the source text.", good="answer vs best source"),
            meter("Answer relevance", a.get("answer_relevance"), "Embedding cosine between the answer and the question.",
                  good="answer vs question"),
            meter("Lexical support", a.get("lexical_support"), "Share of the answer's content words that also appear in the sources.",
                  fmt="{:.0%}", good="answer words found in sources"),
            meter("TF-IDF overlap", a.get("tfidf_answer_context"), "TF-IDF cosine between the answer and all retrieved passages.",
                  good="answer vs sources"),
            meter("Citations", len(valid), "[Source n] references in the answer that point to a retrieved passage.",
                  hi=max(1, n_sources), fmt="{:d}", good=f"{len(valid)} valid of {len(cites)} cited"),
            meter("Lexical diversity", a.get("lexical_diversity"), "Type-token ratio: unique words / total words.",
                  good=f'{a.get("word_count", 0)} words · {a.get("sentence_count", 0)} sentence{"s" if a.get("sentence_count") != 1 else ""}'),
            meter("Semantic mean", a.get("groundedness_mean"), "Average embedding cosine between the answer and all retrieved passages.",
                  good="answer vs all sources"),
        ]
        kw = "".join(f'<span class="grove-term">{esc(k)}</span>' for k in a.get("top_keywords", []))
        st.html('<div class="grove-tiles">' + "".join(tiles) + "</div>"
                + (f'<div class="grove-terms" style="margin-top:10px">Top TF-IDF keywords: {kw}</div>' if kw else ""))


_msg_counter = [0]


def show_message(m):
    _msg_counter[0] += 1
    # keyed container -> CSS class st-key-umsg_N / st-key-amsg_N, used to style user vs assistant bubbles
    box = st.container(key=f"{'umsg' if m['role'] == 'user' else 'amsg'}_{_msg_counter[0]}")
    with box, st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🌳"):
        not_found = m["role"] == "assistant" and is_abstain(m["content"])
        if not_found:
            show_not_found()
        else:
            st.markdown(m["content"])
        if m["role"] == "assistant":
            st.html(meta_html(m.get("mode"), m.get("latency_ms")))
            show_sources(m.get("sources"), not_found)
            show_analysis(m.get("analysis"), len(m.get("sources") or []))


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

# ---------- sidebar ----------
with st.sidebar:
    n_docs = len(engine.documents)
    st.markdown(f'<div class="grove-brand">{LOGO.format(s=34)}<div><b>Grove</b>'
                f'<small>{n_docs} document{"s" if n_docs != 1 else ""} indexed</small></div></div>', unsafe_allow_html=True)
    if st.button("＋  New chat", type="primary", use_container_width=True):
        state.chat_id = None
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
                state.chat_id = c["id"]
                st.rerun()
        with col_del:
            if st.button("✕", key=f"del_{c['id']}", help="Delete chat"):
                db.delete_chat(c["id"])
                if state.chat_id == c["id"]:
                    state.chat_id = None
                st.rerun()

    st.html('<div class="grove-label">Search mode</div>')
    state.mode = st.selectbox("Search mode", MODES, index=MODES.index(state.mode) if state.mode in MODES else 0,
                              format_func=lambda m: MODE_HELP.get(m, (m,))[0], label_visibility="collapsed",
                              help="\n\n".join(f"**{v[0]}**: {v[1]}" for k, v in MODE_HELP.items() if k in MODES))
    st.caption(MODE_HELP.get(state.mode, ("", ""))[1])

    with st.expander(f"📚 Your library ({n_docs})"):
        st.html("".join(f'<div class="grove-doc" title="{esc(d["name"])}"><span>{esc(doc_label(d["name"]))}</span>'
                        f'<em>{d["pages"]} p</em></div>' for d in engine.documents))

# ---------- main ----------
messages = db.get_messages(state.chat_id) if state.chat_id and db.chat_exists(state.chat_id) else []
if state.chat_id and not messages and not db.chat_exists(state.chat_id):
    state.chat_id = None

question = st.chat_input("Ask anything about your documents…")
if state.pending and not question:
    question, state.pending = state.pending, None

if not messages and not question:
    st.markdown(f'<div class="grove-hero">{LOGO.format(s=64)}<h1>What would you like to know?</h1>'
                f'<p>Answers come straight from your PDFs, with the pages they were found on.</p></div>',
                unsafe_allow_html=True)
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

for m in messages:
    show_message(m)

if question:
    question = question.strip()[:2000]
    if state.chat_id is None:
        state.chat_id = db.create_chat(question)
    db.add_message(state.chat_id, "user", question, mode=state.mode)
    show_message({"role": "user", "content": question})

    with st.container(key="amsg_live"), st.chat_message("assistant", avatar="🌳"):
        t0 = time.perf_counter()
        note = st.empty()
        used = {"mode": state.mode, "prep": None}

        def answer_stream():
            """Streams the answer. If the chosen search finds nothing, silently retries with a broader one."""
            modes = fallback_modes(state.mode)
            for n, mode in enumerate(modes):
                if not n:
                    note.caption("🔎 Searching your documents…")
                else:
                    note.caption(f"🔁 Not found with {MODE_HELP[modes[n - 1]][0]} search. "
                                 f"Trying {MODE_HELP[mode][0]} search…")
                prep = engine.prepare(question, mode, messages)
                used.update(mode=mode, prep=prep)
                pieces = engine.stream_answer(question, prep["docs"])
                head = ""
                for piece in pieces:                       # look ahead: is the model abstaining?
                    head += piece
                    if len(head) >= 24:
                        break
                if is_abstain(head) and n < len(modes) - 1:
                    for _ in pieces:
                        pass
                    continue
                note.empty()
                if is_abstain(head):
                    for _ in pieces:
                        pass
                    note.empty()
                    return                                 # shown as a "not found" card below
                yield head
                yield from pieces
                return

        try:
            streamed = st.write_stream(answer_stream())
            answer = streamed if isinstance(streamed, str) else "".join(map(str, streamed or []))
            answer = clean_answer(answer)
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
            sources = engine.sources(prep["docs"]) if prep else []
            try:
                analysis = engine.analyze(question, answer, prep["docs"] if prep else [])
            except Exception:
                analysis = None
            if is_abstain(answer):
                show_not_found()
            st.html(meta_html(mode, latency))
            show_sources(sources, is_abstain(answer))
            show_analysis(analysis, len(sources))
            db.add_message(state.chat_id, "assistant", answer, mode=mode, sources=sources, latency_ms=latency,
                           analysis=analysis)
            st.rerun()   # refresh the sidebar (new chat title) with the saved conversation
