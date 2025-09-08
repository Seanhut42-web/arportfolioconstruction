# Manager Portfolio Analytics (Streamlit)

An interactive Streamlit app to explore manager track records, build custom portfolios, and model FX hedging (GBP base).

## Quick start (local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads the workbook from `data/Manager Track Records v2.xlsx` and caches a Parquet at `data/returns_monthly.parquet` for faster restarts.

## Project layout
```
streamlit-app/
├─ app.py
├─ pages/
│  ├─ 1_Portfolio_Explorer.py
│  └─ 2_Manager_Explorer.py
├─ src/
│  ├─ ingest.py
│  ├─ hedging.py
│  └─ metrics.py
├─ data/
│  └─ Manager Track Records v2.xlsx
├─ requirements.txt
└─ README.md
```

## Notes
- FX sheet is assumed to be the **last sheet** in the workbook; if not, the app tries to auto-detect an FX sheet by scanning for a `USDGBP`/`GBPUSD` rate column and inverts if needed.
- Hedging uses a **CIP proxy**: monthly carry derived from annual GBP/USD cash assumptions. You can blend between **Unhedged (spot)** and **Fully hedged** via the **Hedge ratio** slider.
- Negative weights are allowed, and weights can be normalized to 100%.

## Streamlit Community Cloud
1. Push this folder to a public GitHub repo.
2. Create an app on https://share.streamlit.io/ pointing to `app.py`.
3. Ensure the `data/Manager Track Records v2.xlsx` file is in the repo (or replace it with your own workbook of the same structure).

## Optional: Docker
```Dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build & run:
```bash
docker build -t manager-analytics .
docker run --rm -p 8501:8501 manager-analytics
```
