# LLM Eval Monitoring

Evaluation and monitoring stack for RAG systems. Tracks Ragas metrics over time, blocks deployment on quality regression, and exports AI Act compliance reports.

Works standalone or wired into any RAG API (including [rag_souverain](https://github.com/Sof1ane/rag_souverain)) via `workflow_call`.

---

## Architecture

```
  CUAD corpus
  (511 commercial contracts, HuggingFace)
       │
       ▼
  generate_testset.py                   ← Ragas TestsetGenerator + BGE-M3 (local)
       │  testset.json
       ▼
  run_eval.py ──────────────────────────► RAG API  :8080
       │  question                            │  answer + contexts
       │◄──────────────────────────────────────
       ▼
  Ragas scorer  (LLM-as-judge)
  faithfulness · answer_relevancy · context_precision · context_recall
       │
       ▼
  ┌──────────────────────────────────────────────────┐
  │             eval_postgres  :5433                 │
  │  eval_runs · eval_results · model_registry       │
  │  quality_baselines                               │
  └──────┬───────────────────────────┬───────────────┘
         │                           │
         ▼                           ▼
   Grafana  :3000            export_audit_report.py
   (live trends,             JSON + Markdown
    per-question table,      AI Act Art. 9/12/13
    AI Act view,
    judge cost panels)

CI pipeline:

  push to main
       │
       ▼
  quality_gate.py
  ├─ fetch active baseline from DB
  ├─ run evaluation against live RAG
  ├─ compare metric by metric
  ├─ exit 0  → deployment continues
  └─ exit 1  → BLOCKED  (any metric regressed > 5%)
```

**Stack:** Python 3.11 · Ragas · SQLAlchemy + Alembic · PostgreSQL 16 · Grafana 10 · Docker Compose · GitHub Actions

**Judge LLM:** Claude Haiku by default. Swap to any local OpenAI-compatible endpoint with `JUDGE_BACKEND=openai_compat` — no data leaves the perimeter.

---

## Quick Start

### 1. Install

```bash
git clone <this-repo> llm_eval_monitoring
cd llm_eval_monitoring

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -e .

cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY and RAG_BASE_URL
```

### 2. Start infrastructure

```bash
docker compose up -d
# eval_postgres on :5433, Grafana on :3000
```

### 3. Initialize DB

```bash
alembic upgrade head
```

### 4. Generate testset (one-time)

```bash
# Downloads CUAD from HuggingFace, generates synthetic Q/A with Ragas + BGE-M3
python scripts/generate_testset.py --max-docs 20 --size 30
```

### 5. Capture a baseline

```bash
python scripts/capture_baseline.py --notes "v1 baseline"
```

### 6. Run an evaluation

```bash
python scripts/run_eval.py --model qwen2.5
```

### 7. Dashboard

Grafana: http://localhost:3000 (admin / admin) → folder **RAG Monitoring**

### 8. Audit report

```bash
python scripts/export_audit_report.py --output data/audit_report.json
```

---

## Configuration

See `.env.example` for the full list. Key variables:

| Variable | Default | Notes |
|---|---|---|
| `RAG_BASE_URL` | `http://localhost:8080` | RAG API endpoint |
| `JUDGE_BACKEND` | `anthropic` | `anthropic` or `openai_compat` |
| `ANTHROPIC_API_KEY` | — | Required for `anthropic` backend |
| `JUDGE_BASE_URL` | — | Required for `openai_compat` backend |
| `JUDGE_MODEL` | `claude-haiku-4-5-20251001` | Any model name recognized by the backend |
| `EVAL_POSTGRES_PORT` | `5433` | Separate port to coexist with RAG Postgres on 5432 |
| `GATE_MAX_REGRESSION_DELTA` | `0.05` | Relative regression threshold — fail CI if any metric drops more than 5% |

---

## Sovereign / air-gapped deployment

No external calls required. Set:

```env
JUDGE_BACKEND=openai_compat
JUDGE_BASE_URL=http://localhost:11434/v1   # Ollama, vLLM, LM Studio, etc.
JUDGE_MODEL=mistral
```

Embeddings for testset generation use `BAAI/bge-m3` loaded locally (same model as rag_souverain).

---

## AI Act compliance

Relevant for high-risk AI deployments in banking, insurance, and public sector.

**Article 9 — Risk management:** Quality gates run on every push to `main`. Any regression > 5% on any Ragas metric blocks the pipeline (`exit 1`). Implemented in `ci/quality_gate.py`.

**Article 12 — Record-keeping:** Every run is persisted with full config snapshot (model version, judge model, RAG parameters), per-question scores and latency, judge cost, and CI trigger flag. Model versions are registered in `model_registry` at first evaluation. Migrations tracked with Alembic.

**Article 13 — Transparency:** `scripts/export_audit_report.py` generates a structured JSON + Markdown report on demand, covering all registered models, baselines, and evaluation runs. The Grafana dashboard provides the same visibility in real time.

---

## CI integration with rag_souverain

Add to `rag_souverain/.github/workflows/ci.yml`:

```yaml
jobs:
  eval-gate:
    uses: <org>/llm_eval_monitoring/.github/workflows/rag_souverain_integration.yml@main
    with:
      rag_base_url: http://localhost:8080
      model_version: qwen2.5
    secrets:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

The called workflow spins up its own Postgres, runs `alembic upgrade head`, and executes `quality_gate.py --ci`. Exit 1 blocks the calling repo's deployment.

---

## Project structure

```
llm_eval_monitoring/
├── eval/
│   ├── config.py           # Pydantic Settings — all config from env
│   ├── judge_client.py     # get_judge_llm() — LLM judge, swappable backend
│   ├── judge_cost.py       # cost estimation from token counts
│   ├── metrics.py          # Ragas metric definitions
│   ├── rag_client.py       # RAG API client with retry + timeout
│   ├── runner.py           # full pipeline: testset → RAG → Ragas → DB
│   └── testset/
│       ├── cuad_adapter.py # CUAD dataset → LangChain Documents
│       └── generator.py    # Ragas TestsetGenerator wrapper
├── store/
│   ├── models.py           # SQLAlchemy ORM — 4 tables
│   ├── db.py               # engine + session factory
│   ├── writer.py           # write_eval_run(), write_baseline()
│   └── migrations/         # Alembic versions
├── scripts/
│   ├── generate_testset.py # CLI: generate and save testset.json
│   ├── run_eval.py         # CLI: run evaluation, print results
│   ├── capture_baseline.py # CLI: pin current scores as active baseline
│   └── export_audit_report.py  # CLI: export AI Act audit report
├── ci/
│   └── quality_gate.py     # CI gate — exit 0/1/2
├── dashboard/
│   └── grafana/
│       ├── provisioning/   # auto-provisioned datasource + folder
│       └── dashboards/     # rag_quality.json
├── .github/workflows/
│   ├── eval_ci.yml                    # lint + unit tests + quality gate
│   └── rag_souverain_integration.yml  # workflow_call entrypoint
├── docker-compose.yml      # eval_postgres + Grafana
├── pyproject.toml
└── .env.example
```
