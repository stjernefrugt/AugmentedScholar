# AugmentedScholar

A **Scientific Second Brain** that maps your citation ecosystem into an interactive 3D knowledge graph. Give it your author profile; it builds a local library of PDFs and metadata, computes network centrality, and renders a navigable 3D visualization.

---

## Features

- **Bibliographic expansion** — seed from Semantic Scholar or Google Scholar, traverse references and citations up to 2 hops
- **PDF acquisition** — Unpaywall (free OA) → missing-PDF manifest → Playwright batch downloader with institute SSO support
- **Metadata enrichment** — fetches abstracts, citation counts, and keywords for any paper missing them
- **ISO 4 filenames** — journal names abbreviated via pyiso4 / LTWA standard (`nat-commun`, `phys-rev-lett`)
- **Network analysis** — betweenness and eigenvector centrality, gap analysis (isolated nodes, high-centrality papers without PDFs, missing abstracts)
- **3D visualization** — interactive Plotly Scatter3d dashboard with distinct marker shapes per paper type, blue→red year colorscale, and click-to-inspect paper cards

---

## Folder structure

```
AugmentedScholar/
├── src/
│   ├── ss_client.py          # Semantic Scholar Graph API client (exponential backoff)
│   ├── gscholar_client.py    # Google Scholar scraper (scholarly)
│   ├── unpaywall_client.py   # Unpaywall OA resolver
│   ├── models.py             # Pydantic data models (PaperMetadata, Author, …)
│   ├── citation_models.py    # CitationRecord, ExpansionResult, RelationshipType
│   ├── citation_explorer.py  # BFS expansion engine (Tier 0/1/2)
│   ├── ingest.py             # PDF download + JSON sidecar writer
│   ├── enrichment.py         # Post-expansion abstract/keyword fetcher
│   ├── naming.py             # pyiso4-based filename stem builder
│   ├── metadata_fetcher.py   # DOI → metadata resolver
│   ├── pdf_handler.py        # PDF utilities
│   ├── graph.py              # NetworkX CitationGraph (load/save/PyVis export)
│   ├── graph_builder.py      # Centrality computation + gap analysis
│   └── viz_engine.py         # Plotly 3D figure builder
├── tests/                    # pytest suite (172 tests)
├── run_expansion.py          # CLI entry point — expand and ingest
├── download_pdfs.py          # Playwright-assisted PDF batch downloader
├── migrate_filenames.py      # Rename existing library to new ISO 4 convention
├── app.py                    # Streamlit dashboard
├── requirements.txt
└── library/                  # Created at runtime
    ├── <stem>.pdf
    ├── <stem>.json           # Metadata sidecar per paper
    ├── citation_graph.json   # Persisted NetworkX graph
    └── missing_pdfs.json     # Manifest for download_pdfs.py
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# For Playwright-assisted PDF downloads (optional):
pip install playwright
playwright install chromium
```

---

## Usage

### 1 — Find your Semantic Scholar author ID

```bash
python run_expansion.py --find-author "Your Name"
```

### 2 — Build your library

**From Semantic Scholar** (fast, ~1 req/s unauthenticated):
```bash
python run_expansion.py --author-id 1741101 --library-dir ./library
```

**From Google Scholar** (more complete Tier 0, recommended):
```bash
python run_expansion.py \
    --gscholar-url "https://scholar.google.com/citations?user=AbCdEfGhIjK" \
    --library-dir ./library \
    --unpaywall-email your@email.com
```

**Full two-hop expansion** (author → references → their references):
```bash
python run_expansion.py \
    --author-id 1741101 \
    --depth 2 \
    --api-key YOUR_SS_KEY \
    --library-dir ./library \
    --unpaywall-email your@email.com \
    --enrich
```

| Flag | Default | Description |
|---|---|---|
| `--depth` | `1` | `0` = own papers only, `1` = + direct refs/cites, `2` = + one further hop |
| `--api-key` | — | Semantic Scholar API key; raises rate limit from ~1 to ~10 req/s |
| `--unpaywall-email` | — | Enables Unpaywall OA resolver before adding to missing-PDF manifest |
| `--enrich` | off | Fetch missing abstracts/keywords after expansion (auto-on at `--depth 0`) |
| `--enrich-delay` | `1.0` | Seconds between enrichment requests |

### 3 — Download missing PDFs

```bash
# Generate manifest (done automatically by run_expansion.py):
# library/missing_pdfs.json

# Open browser, log in to your institute once, then auto-download:
python download_pdfs.py --manifest ./library/missing_pdfs.json

# Already on institute VPN — skip the login pause:
python download_pdfs.py --manifest ./library/missing_pdfs.json --no-login-pause
```

### 4 — Launch the dashboard

```bash
streamlit run app.py -- --library-dir ./library --graph-path ./library/citation_graph.json
```

Open `http://localhost:8501`.

**Citation Map tab** — 3D network with toggles for Own / Cited / Citing papers. Click any node for a paper card (title, authors, abstract, keywords, DOI link).

**Gap Analysis tab** — isolated nodes, high-centrality papers without PDFs, papers missing abstracts.

**Analytics tab** — centrality leaderboard sorted by betweenness.

### 5 — Migrate existing library filenames (after first run)

```bash
# Dry run first:
python migrate_filenames.py --library-dir ./library

# Apply:
python migrate_filenames.py --library-dir ./library --apply
```

---

## API dependencies

| Service | Used for | Auth | Rate limit |
|---|---|---|---|
| [Semantic Scholar Graph API v1](https://api.semanticscholar.org/graph/v1) | Paper search, author papers, references, citations | Optional API key | ~1 req/s (unauth) / ~10 req/s (key) |
| [Google Scholar](https://scholar.google.com) | Tier 0 author paper list | None (scraped via `scholarly`) | Throttled automatically |
| [Unpaywall](https://unpaywall.org/products/api) | Open-access PDF URL lookup | Email address (free) | 100k req/day |

The Semantic Scholar client retries automatically on `429 / 500 / 502 / 503 / 504` with exponential backoff (1 s → 2 s → 4 s … capped at 60 s, up to 5 retries).

**Semantic Scholar endpoints used:**

```
/author/search
/paper/search
/author/{id}/papers
/paper/{id}/references
/paper/{id}/citations
```

---

## Development

```bash
# Lint + format
python -m ruff check src/ tests/
python -m ruff format src/ tests/

# Type checking
python -m mypy src/

# Tests
python -m pytest
```
