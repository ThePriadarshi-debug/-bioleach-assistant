# Bioleaching literature assistant

A question-answering assistant over open-access research papers on **bioleaching of waste printed circuit boards (PCBs) and e-waste**. Every claim in an answer is cited and linked to its source paper by DOI.

## How it works
1. Papers were selected by licence: only articles published under **CC BY 4.0** (checked via Unpaywall) are included.
2. Text was extracted, reference sections removed, and split into passages.
3. Passages were embedded with `BAAI/bge-small-en-v1.5`; a question retrieves the closest passages.
4. A language model (via Groq) answers **only** from those passages and cites them.

## Licences and attribution
The papers remain the copyright of their authors and are used under their CC BY 4.0 licences. The list of included papers, with DOIs, is in `index/papers.csv` and in the app's sidebar.

AI-generated answers can be wrong. Always check the cited paper.

Built by Pushpraj Priadarshi.
