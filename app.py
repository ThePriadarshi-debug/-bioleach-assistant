"""Bioleaching literature assistant (public demo).

Answers questions from open-access (CC BY 4.0) papers on bioleaching of waste
printed circuit boards and e-waste, citing every claim with its source and DOI.

Local run:   $env:GROQ_API_KEY = Get-Content P:\\bioleach-rag\\groq_key.txt
             streamlit run app.py
Cloud:       put GROQ_API_KEY in the app's Secrets on Streamlit Community Cloud.
"""
import csv, json, os, re, time
from collections import deque
from pathlib import Path

import numpy as np
import streamlit as st

# ------------------------------------------------------------------ settings
INDEX = Path(__file__).parent / "index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
TOP_K = 5
MIN_SIMILARITY = 0.55          # passages less similar than this are ignored
MAX_QUESTIONS_PER_VISITOR = 15
MAX_QUESTIONS_PER_MINUTE = 4   # shared by all visitors; protects the free Groq quota
BUILT_BY = "Pushpraj Priadarshi"

EXAMPLES = [
    "What microbes are used to leach copper from PCBs?",
    "How does pulp density affect metal recovery in bioleaching?",
    "What copper recovery percentages have been reported, and under what conditions?",
    "How is gold recovered from e-waste by bioleaching?",
]

SYSTEM = """You are a careful research assistant on bioleaching of waste printed circuit boards (PCBs = printed circuit boards) and e-waste.

How to answer:
1. Read EVERY numbered source and decide whether it is relevant to the question.
2. Use ALL relevant sources, not just the first one.
3. Group findings logically (by organism, metal or process variable) as short bullet points.
4. Cite every bullet with its source number, like [3] or [1, 4].
5. Keep separate studies separate; never merge two studies into one claim.
6. Keep numbers with their conditions (pulp density, time, temperature, organism).
7. If sources disagree, say so.
8. Use ONLY the sources. If none answer the question, reply exactly: I could not find this in the papers.
Do not add a reference list at the end; the app shows the sources."""

# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner="Loading the paper index...")
def load_index():
    chunks = [json.loads(l) for l in open(INDEX / "chunks.jsonl", encoding="utf-8")]
    vecs = np.load(INDEX / "embeddings.npy").astype(np.float32)
    with open(INDEX / "papers.csv", encoding="utf-8") as f:
        papers = list(csv.DictReader(f))
    return chunks, vecs, papers


@st.cache_resource(show_spinner="Loading the search model (first visit takes about a minute)...")
def load_embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(EMBED_MODEL)


@st.cache_resource
def rate_window():
    return deque()                # timestamps of recent questions, shared by all visitors


def api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


def models():
    try:
        first = st.secrets.get("GROQ_MODEL")
    except Exception:
        first = os.environ.get("GROQ_MODEL")
    return ([first] if first else []) + [m for m in DEFAULT_MODELS if m != first]

# ------------------------------------------------------------------ logic
def retrieve(question, chunks, vecs):
    q = np.array(list(load_embedder().query_embed([question])))[0]
    q /= np.linalg.norm(q)
    sims = vecs @ q
    hits = []
    for i in np.argsort(sims)[::-1][:TOP_K]:
        if sims[i] >= MIN_SIMILARITY:
            hits.append({**chunks[i], "sim": float(sims[i])})
    return hits


def stream_answer(question, hits):
    from groq import Groq
    client = Groq(api_key=api_key())
    context = "\n\n".join(f"[{i}] ({h['title'][:90]}, {h['year']}, page {h['page']})\n{h['text']}"
                          for i, h in enumerate(hits, 1))
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}]
    last_error = None
    for model in models():
        try:
            stream = client.chat.completions.create(model=model, messages=messages, temperature=0,
                                                    max_tokens=900, stream=True)
            st.session_state.model_used = model
            for part in stream:
                delta = part.choices[0].delta.content
                if delta:
                    yield delta
            return
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            if "rate" in msg or "429" in msg:
                raise RuntimeError("busy") from e
            continue                  # model retired or unavailable: try the next one
    raise RuntimeError(f"No model available: {last_error}")


def cited(answer, n):
    used = set()
    for g in re.findall(r"\[([\d,\s]+)\]", answer):
        used |= {int(x) for x in re.findall(r"\d+", g)}
    return used & set(range(1, n + 1))


def show_sources(hits, used):
    if not hits:
        return
    with st.expander(f"Sources: {len(used)} of {len(hits)} cited"):
        for i, h in enumerate(hits, 1):
            with st.container(border=True):
                tag = "cited" if i in used else "not used"
                st.markdown(f"**[{i}] {h['title']}**  \n"
                            f"{h['journal']} ({h['year']}), page {h['page']} · {h['license']} · `{tag}`")
                st.caption(h["text"][:350] + " ...")
                if h.get("doi"):
                    st.markdown(f"[Read the paper: doi.org/{h['doi']}](https://doi.org/{h['doi']})")

# ------------------------------------------------------------------ page
st.set_page_config(page_title="Bioleaching literature assistant", page_icon=":material/biotech:", layout="wide")
st.markdown("""<style>
.hero {padding:1.2rem 1.5rem;border-radius:16px;margin-bottom:1rem;
       background:linear-gradient(115deg,#1d4a37 0%,#2c6b50 55%,#b8692e 100%);}
.hero h1 {color:#fff;margin:0;font-size:1.9rem;}
.hero p {color:#eef4f0;margin:.35rem 0 0 0;}
</style>""", unsafe_allow_html=True)

chunks, vecs, papers = load_index()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("asked", 0)

with st.sidebar:
    st.subheader(":material/info: About")
    st.markdown(f"Answers come only from **{len(papers)} open-access papers** (CC BY 4.0) on bioleaching "
                "of printed circuit boards and e-waste. Every claim is cited and linked to its source.")
    st.markdown(f"Questions left this visit: **{MAX_QUESTIONS_PER_VISITOR - st.session_state.asked}**")
    if st.button("Clear chat", icon=":material/delete:"):
        st.session_state.messages = []
        st.rerun()
    with st.expander(f"The {len(papers)} papers"):
        for p in sorted(papers, key=lambda p: p["year"] or "", reverse=True):
            st.markdown(f"- [{p['title']}](https://doi.org/{p['doi']}) — {p['journal']}, {p['year']}")
    st.caption("AI answers can be wrong. Always check the cited paper before relying on a claim. "
               f"Built by {BUILT_BY}. Papers are used under their CC BY 4.0 licences; "
               "copyright remains with the authors.")

st.markdown(f"""<div class="hero"><h1>Bioleaching literature assistant</h1>
<p>Ask about bioleaching of waste printed circuit boards. Answers cite {len(papers)} open-access papers.</p></div>""",
            unsafe_allow_html=True)

if not api_key():
    st.error("The app owner has not set GROQ_API_KEY yet.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/biotech:"):
        st.markdown(msg["content"])
        if msg.get("caption"):
            st.caption(msg["caption"])
        show_sources(msg.get("hits", []), msg.get("used", set()))

if not st.session_state.messages:
    st.markdown("##### Try a question")
    cols = st.columns(2)
    for j, q in enumerate(EXAMPLES):
        if cols[j % 2].button(q, key=f"ex{j}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()

question = st.chat_input("Ask a question about bioleaching of PCBs...") or st.session_state.pop("pending", None)
if question:
    question = question[:500]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/biotech:"):
        window, now = rate_window(), time.time()
        while window and now - window[0] > 60:
            window.popleft()
        hits, used, caption = [], set(), ""
        if st.session_state.asked >= MAX_QUESTIONS_PER_VISITOR:
            answer = "You have reached the question limit for this visit. Please come back later."
            st.markdown(answer)
        elif len(window) >= MAX_QUESTIONS_PER_MINUTE:
            answer = "The free demo is busy right now. Please try again in a minute."
            st.markdown(answer)
        else:
            with st.spinner("Searching the papers..."):
                hits = retrieve(question, chunks, vecs)
            if not hits:
                answer = "I could not find this in the papers."
                st.markdown(answer)
                st.caption("No passage was close enough. Try rephrasing your question.")
            else:
                window.append(now)
                st.session_state.asked += 1
                t0 = time.time()
                try:
                    answer = st.write_stream(stream_answer(question, hits))
                    used = cited(answer, len(hits))
                    caption = (f"{time.time() - t0:.0f} s · cited {len(used)} of {len(hits)} sources · "
                               f"{st.session_state.get('model_used', '')}")
                    st.caption(caption)
                except RuntimeError as e:
                    answer = ("The free demo is busy right now. Please try again in a minute."
                              if str(e) == "busy" else "Sorry, the answer service is unavailable right now.")
                    st.markdown(answer)
            show_sources(hits, used)
    st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits,
                                      "used": used, "caption": caption})
    st.rerun()
