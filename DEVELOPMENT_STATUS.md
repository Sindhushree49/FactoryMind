# FactoryMind - Development Status

**Last updated:** 2026-10-03 (session 1)
**Phase:** 1 of 12 - Workspace audit and project structure

---

## 1. Workspace Audit Result

The prompt assumed `FactoryMind` already contained a React frontend, a Python
backend, ML models and datasets. **This was not the case.** Verified findings:

| Item | Expected by prompt | Actual state |
| --- | --- | --- |
| Frontend (React/TS/Vite) | Existing | **Absent** - zero files |
| Backend (FastAPI) | Existing | **Absent** - zero files |
| ML models | Existing | **Absent** |
| Datasets (AI4I / C-MAPSS / MVTec) | Existing | **Absent** |
| `package.json`, `requirements.txt` | Existing | **Absent** |

`System.IO.Directory.GetFileSystemEntries()` on
`C:\Users\Ravi9\OneDrive\Desktop\FactoryMind` returned **0 entries**. The
directory existed but was completely empty.

Consequence: nothing could be "preserved" because there was nothing to
preserve. The entire platform is being built from scratch. The directory
layout in the specification was followed as the target structure.

### Detected toolchain

| Tool | Version | Note |
| --- | --- | --- |
| Python | 3.11.9 (`Python311`) | Used for backend - matches spec |
| Python | 3.14.3 (default) | Not used for backend (ML wheel support) |
| Node.js | 25.7.0 | Used for frontend |
| npm | 11.10.1 | - |
| Git | 2.46.0 | Repo not yet initialised |
| Docker | **Not installed** | Docker artefacts written but untested |
| PostgreSQL | **Not installed** | SQLite dev fallback + Postgres in compose |

---

## 2. Completed

- [x] Workspace audit (empty-directory discovery documented above)
- [x] Repository directory tree created (36 directories)
- [x] `.gitignore` covering datasets, model binaries, node_modules, secrets
- [x] `.env.example` with documented defaults and safe SQLite fallback
- [x] `backend/requirements.txt` with pinned, Python 3.11-compatible versions
- [x] Python 3.11 virtual environment created at `.venv`
- [x] `backend/scripts/download_datasets.py` - provenance-recording downloader
- [x] Network reachability verified for dataset hosts

## 3. In Progress

- [ ] `pip install -r backend/requirements.txt` (running in background)
- [ ] Dataset acquisition: AI4I 2020, NASA C-MAPSS

## 4. Pending

- [ ] Phase 4 - Backend core (config, logging, SQLAlchemy models, schemas)
- [ ] Phase 5 - AI4I failure prediction (Random Forest + XGBoost)
- [ ] Phase 6 - Anomaly detection (Isolation Forest)
- [ ] Phase 7 - NASA C-MAPSS RUL regression (XGBoost baseline + LSTM)
- [ ] Phase 8 - MVTec AD computer vision inspection
- [ ] Phase 9 - SHAP explainability service
- [ ] Phase 10 - FastAPI routers (18 specified endpoints)
- [ ] Phase 11 - RAG knowledge base
- [ ] Phase 12 - LangGraph six-agent system
- [ ] Phase 13 - React + TS + Vite + Tailwind frontend (12 pages)
- [ ] Phase 14 - Test suite execution
- [ ] Phase 15 - Docker, Alembic, README, Mermaid architecture

## 5. Blocked

| Blocker | Impact | Mitigation |
| --- | --- | --- |
| **MVTec AD has no unauthenticated download URL.** `mvtec.com/.../mvtec_ad.zip` returns HTTP 404 because MVTec requires explicit licence acceptance before issuing the archive. | Prevents training/evaluating the visual inspection model on real images. | Downloader raises a clear blocker with manual steps. The CV module detects the absence of data, returns `model_trained: false`, and refuses to fabricate anomaly scores. Manual instructions: see section below. |
| **No Docker daemon** | `docker-compose.yml` cannot be executed/verified here. | Compose file still authored; documented as unverified. |
| **No PostgreSQL server** | Cannot verify Postgres dialect at runtime. | App supports both dialects; SQLite used for the executed test run. Postgres path covered by Docker Compose. |
| **No `OPENAI_API_KEY`** | LLM-backed agent reasoning unavailable. | Deterministic rule-based fallback engine implemented and clearly labelled in every agent response (`engine: rule_based`). |

### MVTec AD manual setup

1. Open <https://www.mvtec.com/company/research/datasets/mvtec-ad>
2. Accept the CC BY-NC-SA 4.0 non-commercial licence and download
   `mvtec_anomaly_detection.tar.xz`.
3. Extract into `datasets/mvtec_ad/` so the 15 category folders sit at
   `datasets/mvtec_ad/mvtec_anomaly_detection/{bottle, cable, capsule, carpet,
   grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush,
   transistor, wood, zipper}`.
4. Re-run `python backend/scripts/download_datasets.py --dataset mvtec --verify`
   and then `python backend/app/ml/computer_vision/train.py`.

---

## 6. Testing Results

**No tests have been executed yet.** This section will only ever record results
from actual runs. See Phase 14.

---

## 7. Exact Next Task

Install completion check, then Phase 4 backend core:

```powershell
# 1. Confirm the dependency install succeeded
.\.venv\Scripts\python.exe -c "import fastapi, sklearn, xgboost, shap, torch, cv2, langgraph, faiss; print('deps OK')"

# 2. Confirm the datasets landed
Get-Content datasets\manifests\datasets.json

# 3. Then implement, in order:
#    backend/app/core/config.py
#    backend/app/core/logging_config.py
#    backend/app/database/base.py, session.py
#    backend/app/models/*.py            (13 tables)
#    backend/app/schemas/*.py
```
