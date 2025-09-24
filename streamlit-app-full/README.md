# Manager Portfolio Analytics (Streamlit)

This app parses a multi‑sheet Excel workbook of manager track records, monthlyises mixed/irregular frequency series, converts USD managers to GBP either **unhedged (spot)** or **fully hedged (CIP proxy)** with a **hedge ratio** blend, and lets you build custom portfolios.

---
## What’s new (Updated 2025-09-24)
- **Monthly Bars → Distribution** sub‑tab: Histogram + KDE (with FD/Scott/Sturges binning or fixed), Violin + Box, ECDF, Q–Q, summary stats (mean/std/skew/kurtosis, % positive, best/worst with dates, VaR/ES 95/99, Sharpe), and CSV exports.
- **Weight sliders** now constrain to **0 → 1** with normalisation toggle.
- **Detailed PDF report** (Matplotlib/Seaborn + PyMuPDF) that mirrors the Explorer analyses (Cumulative, Drawdown, 12M Rolling, Bars, Distribution, Correlation, Year×Month, Contributions). Falls back to your original `src.report.build_pdf` if PyMuPDF isn't present.

---
## Repo layout
```
.
├─ 1_Portfolio_Explorer.py          # Main interactive page (can be run standalone)
├─ src/
│  ├─ __init__.py
│  ├─ ingest.py                      # workbook → monthly GBP returns
│  ├─ hedging.py                     # hedging inputs & panel builder
│  ├─ metrics.py                     # drawdown + summary stats
│  ├─ contrib.py                     # return/risk contributions
│  └─ state.py                       # theme + URL state
├─ data/
│  └─ Manager Track Records v2.xlsx
├─ requirements.txt
└─ README.md
```
> If you use a multipage Streamlit app with a landing `app.py`, place `1_Portfolio_Explorer.py` under `pages/` and run `streamlit run app.py`.

---
## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scriptsctivate
pip install -r requirements.txt
```

## Run
```bash
# Option A: run the portfolio explorer directly
streamlit run 1_Portfolio_Explorer.py

# Option B (if you have a multipage app):
streamlit run app.py
```

### Data
Place your workbook at:
```
./data/Manager Track Records v2.xlsx
```

---
## Features overview
- Manager selection, **0–1 weights** with optional **normalise to 100%**
- **FX handling**: Unhedged (spot) vs Fully hedged (CIP proxy) with **hedge ratio**
- Summary, Cumulative, Drawdown, **12M Return**, **12M Vol**, **Monthly Bars** (+ **Distribution**), **Correlation**, **Year×Month**
- **Dark mode** toggle; charts adopt the theme
- **Download PDF report** (detailed). Requires `PyMuPDF` (`import fitz`).

---
## PDF Report
The app generates a multi‑page PDF via **PyMuPDF** from static **Matplotlib/Seaborn** figures. If `PyMuPDF` is not installed, it attempts to use your legacy `src.report.build_pdf`.

---
## Notes & Tips
- Click **Run portfolio analytics** once; all charts reuse computed results.
- If you update the Excel, use **Reload data** in the sidebar to refresh cached inputs.
- The Distribution tab includes **winsorize 1% tails** and **binning** controls.

---
## Troubleshooting
- If PDF generation fails, ensure `PyMuPDF` is installed and that the workbook path exists.
- For headless servers: `streamlit run 1_Portfolio_Explorer.py --server.headless true`.

---
## License
Internal use.
