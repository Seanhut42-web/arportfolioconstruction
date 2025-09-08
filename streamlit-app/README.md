# AR Portfolio Construction (Streamlit)

A self-contained Streamlit app with:

- **Portfolio Explorer** (heatmap intact)
- **Manager Explorer**
- **Factor Attribution** (additive; uses Statsmodels for OLS / HAC)

## Quick start

```bash
pip install -r streamlit-app/requirements.txt
streamlit run streamlit-app/app.py
```

## Data
- Demo manager returns at `streamlit-app/data/demo/demo_managers.csv`
- Factors at `streamlit-app/data/factors/Factor_Returns_standardized.csv`

You can replace these with your files via the UI.
