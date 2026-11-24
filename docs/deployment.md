# Deploying a public demo

## The decision that matters: don't put an API key behind a public URL

A hosted demo wired to an Anthropic key is a **public spend endpoint**. Anyone
who finds the URL can run generations against it, billed to the key's owner,
with no cap and no attribution. Scrapers find Streamlit Cloud URLs.

So the public demo runs **retrieval-only, with no API key configured at all**:

| | Public demo | Local, with your key |
|---|---|---|
| Semantic search over the full corpus | yes | yes |
| Passages, scores, page numbers, citations metadata | yes | yes |
| Document library, filters, upload | yes | yes |
| Claude-generated grounded answers | **no** | yes |
| Can a visitor spend money? | **no — there is nothing to spend** | n/a |

This is not a crippled demo. Retrieval *is* the interesting half of a RAG
system, and it is the half where quality is decided. The app detects the absent
key and says so plainly in the sidebar rather than failing at click time.

If you later want generation in a hosted demo, do **not** simply add the key.
Put authentication in front of it, cap spend with a separate key used nowhere
else, and rate-limit per session.

## Streamlit Community Cloud (free)

The repo is code-only — PDFs and the vector index are gitignored. Set
`REGINTEL_BOOTSTRAP=1` and the app builds its own corpus on first start:
it downloads the 21 documents in `data/sources.json` (all from hosts that permit
automated access — govinfo, FDIC, FinCEN) and indexes them locally. No
credentials, no manual step.

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** → pick the `regintel-ai` repo, branch `main`.
3. Main file path: `app/streamlit_app.py`
4. **Advanced settings → Secrets**, paste exactly this and nothing else:

   ```toml
   REGINTEL_BOOTSTRAP = "1"
   ```

   **Do not add `ANTHROPIC_API_KEY` here.** That is the whole point.
5. Deploy.

First boot takes a few minutes: it installs dependencies (including torch),
downloads the embedding model (~90 MB) and the 21 PDFs, then embeds 229 pages.
Later boots reuse the cached ChromaDB index.

### If the first boot is too slow for the free tier

Trim `data/sources.json` to a handful of documents — the two CFR parts plus
three or four FFIEC sections still demonstrate cross-document retrieval and
both metadata filters.

## Running it locally with generation

```bash
cp .env.example .env          # then put your real key in .env
python scripts/fetch_sources.py
python -m regintel.cli ingest data/pdfs
streamlit run app/streamlit_app.py
```

`.env` is gitignored. Keep the key there and nowhere else — not in
`.streamlit/secrets.toml`, not in a committed file, not in a shell history line
you later paste somewhere.

## What is safe to make public

Already true of this repository:

- **No secrets committed.** `.env`, `*.key` and `.streamlit/secrets.toml` are
  gitignored; only `.env.example` with a placeholder is tracked.
- **No corpus committed.** PDFs and the index are gitignored and rebuilt from the
  manifest, so the repo stays small and carries no redistribution question.
- **CI needs no secrets** and asserts that `ANTHROPIC_API_KEY` is absent.

Before any deploy, confirm nothing sensitive is staged:

```bash
git ls-files | grep -iE '\.env$|secrets|\.key$'   # expect no output
```
