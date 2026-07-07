# RS Course Project — MovieLens Recommender Dashboard

ITI AI-Pro Intake 46 — Recommender Systems course project. Factorization Machine
recommender + item similarity, served via FastAPI and a Streamlit dashboard.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Download MovieLens (small) into `data/raw/` from
[grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/).

## Running

**1. Train the model** (produces `models/{fm,deepfm}_model.pt`, matching `_config.json`,
`models/item_embeddings.npy`):

```bash
jupyter notebook notebooks/03_train_fm_model.ipynb       # baseline FM
jupyter notebook notebooks/04_train_deepfm_model.ipynb   # DeepFM (FM + deep MLP)
```

`backend/config.yaml` points at whichever model's `.pt`/`.json` files you want served —
swap `fm_model_path`/`fm_config_path` to the DeepFM artifacts once trained. The config's
`model_type` field ("fm" or "deepfm") tells `backend/models/fm_inference.py` which class to load.

**2. Start the backend:**

```bash
uvicorn backend.main:app --reload --port 8000
```

**3. Start the dashboard:**

**Option A: New Dash dashboard (recommended — rich UI with charts and filters):**
```bash
python dashboard/dash_app.py
```
Dashboard at `http://localhost:8050`

**Option B: Original Streamlit dashboard (simple):**
```bash
streamlit run dashboard/app.py
```

## Project structure

```
data/            raw + processed MovieLens data (gitignored)
notebooks/       EDA, preprocessing, training, similarity notebooks
models/          saved inference artifacts (gitignored)
src/             shared code — model class + data utils, imported by both
                 notebooks and backend to keep train/inference consistent
backend/         FastAPI app (user recs + item similarity endpoints)
dashboard/       Streamlit UI
tests/           pytest tests for backend endpoints
```

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /recommend/user/{user_id}?page=&page_size=` | Top-N recommendations for a user |
| `GET /similar/item/{item_id}?page=&page_size=` | Top-N similar items |
| `GET /health` | Health check |

## Notes

- The FM model class lives in `src/fm_model.py` and is imported by both the
  training notebook and `backend/models/fm_inference.py` — don't redefine it
  in the notebook, or saved weights won't load cleanly.
- `DeepFM` (same file) extends `FactorizationMachine` — it wraps an FM instance
  internally and adds an MLP tower on the same embeddings, so both models share
  one inference loader. Set `model_type` in `fm_config.json` to switch between them.
- `backend/config.yaml` is the single source of truth for model paths and
  pagination defaults.
