"""Bioleaching literature assistant (public demo).

Answers questions from open-access (CC BY 4.0) papers on bioleaching of waste
printed circuit boards and e-waste, citing every claim with its source and DOI.

Local run:   $env:GROQ_API_KEY = Get-Content P:\\bioleach-rag\\groq_key.txt
             streamlit run app.py
Cloud:       put GROQ_API_KEY in the app's Secrets on Streamlit Community Cloud.
"""
import csv, json, math, os, re, time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import streamlit as st

# ------------------------------------------------------------------ settings
HERE = Path(__file__).parent
# index files may sit in index/ or, if uploaded through the GitHub website, next to app.py
INDEX = HERE / "index" if (HERE / "index" / "chunks.jsonl").exists() else HERE
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]   # llama-3.x models were retired by Groq in Aug 2026
TOP_K = 6                      # passages sent to the model (kept modest for the free Groq quota)
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
You have NO tools and NO browser: never call a tool, just write the answer as text.

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

# ------------------------------------------------------------------ smarter search
# 1. the model rewrites the question into the technical terms a paper would use
# 2. meaning search (bge-small) and keyword search (BM25) run for the question AND the rewrite
# 3. the four ranked lists are merged with Reciprocal Rank Fusion
# Tested on 20 reworded expert questions: right paper in the top 5 rose from 40% to 65%.
REWRITE_PROMPT = ("You help search a library of research papers on bioleaching of waste printed circuit boards "
                  "(PCBs = printed circuit boards, not polychlorinated biphenyls). Rewrite the user's question as "
                  "ONE short search query using the technical terms such a paper would use (process names, organism "
                  "groups, reactor types, measured quantities). Do not answer it. Return ONLY the query.")
STOP = set("a an the of and or in on to for with by from is are was were be been can could does do did "
           "has have had it its this that these those which what how why when where who as at into than "
           "there their they any some other more most such using used use based via".split())


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1]


@st.cache_resource(show_spinner=False)
def keyword_index():
    chunks, _, _ = load_index()
    toks = [tokens(c.get("title", "") + " " + c["text"]) for c in chunks]
    lens = np.array([len(t) for t in toks], dtype=np.float32)
    postings = defaultdict(list)
    for i, t in enumerate(toks):
        for w, tf in Counter(t).items():
            postings[w].append((i, tf))
    n = len(toks)
    idf = {w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in postings.items()}
    return postings, idf, lens, float(lens.mean())


def bm25(query, n, k1=1.5, b=0.75):
    postings, idf, lens, avg = keyword_index()
    s = np.zeros(n, dtype=np.float32)
    for w in set(tokens(query)):
        for i, tf in postings.get(w, []):
            s[i] += idf[w] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lens[i] / avg))
    return s


def rewrite_query(question):
    """One small Groq call; if anything fails, just search the question as typed."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key())
        for model in models():
            try:
                r = client.chat.completions.create(model=model, temperature=0, max_tokens=120, messages=[
                    {"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": question}])
                text = normalize((r.choices[0].message.content or "").strip().strip('"').splitlines()[0]).strip(" 】")
                return text if 5 < len(text) < 400 else question
            except Exception as e:
                if ("rate" in str(e).lower() or "429" in str(e)) and "per day" not in str(e).lower():
                    return question
    except Exception:
        pass
    return question


def retrieve(question, chunks, vecs, search_q=None):
    queries = [question] + ([search_q] if search_q and search_q != question else [])
    best_sim = np.full(len(chunks), -1.0, dtype=np.float32)
    rankings = []
    for q in queries:
        v = np.array(list(load_embedder().query_embed([q])))[0]
        sims = vecs @ (v / np.linalg.norm(v))
        best_sim = np.maximum(best_sim, sims)
        rankings.append(np.argsort(-sims)[:200])
        kw = bm25(q, len(chunks))
        top = np.argsort(-kw)[:200]
        rankings.append(top[kw[top] > 0])
    score = defaultdict(float)
    for r in rankings:
        for pos, i in enumerate(r):
            score[i] += 1 / (60 + pos + 1)
    best = sorted(score, key=score.get, reverse=True)[:TOP_K]
    return [{**chunks[i], "sim": float(best_sim[i])} for i in best if best_sim[i] >= MIN_SIMILARITY]


def stream_answer(question, hits):
    from groq import Groq
    client = Groq(api_key=api_key())
    context = "\n\n".join(f"[{i}] ({h['title'][:90]}, {h['year']}, page {h['page']})\n{h['text']}"
                          for i, h in enumerate(hits, 1))
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}]
    last_error = None
    for model in models():
        for attempt in range(3):      # gpt-oss sometimes tries a built-in 'browser' tool; just retry
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
                if "tool_use_failed" in msg or "called a tool" in msg:
                    continue
                if ("rate" in msg or "429" in msg) and "per day" not in msg:
                    raise RuntimeError("busy") from e
                break                 # retired, out of daily quota or unavailable: try the next model
    raise RuntimeError(f"No model available: {last_error}")


def normalize(answer):
    """Some models (e.g. gpt-oss) cite with full-width brackets like 【1】; convert to [1]."""
    return re.sub(r"\s*【\s*([\d,\s]+?)\s*】", r" [\1]", answer)


def cited(answer, n):
    used = set()
    answer = normalize(answer)
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
            with st.spinner("Understanding your question..."):
                search_q = rewrite_query(question)
            if search_q != question:
                st.caption(f"Searching for: {search_q}")
            with st.spinner("Searching the papers..."):
                hits = retrieve(question, chunks, vecs, search_q)
            if not hits:
                answer = "I could not find this in the papers."
                st.markdown(answer)
                st.caption("No passage was close enough. Try rephrasing your question.")
            else:
                window.append(now)
                st.session_state.asked += 1
                t0 = time.time()
                try:
                    answer = normalize(st.write_stream(stream_answer(question, hits)))
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
