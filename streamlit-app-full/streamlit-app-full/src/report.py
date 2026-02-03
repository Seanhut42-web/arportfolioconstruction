from __future__ import annotations
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fitz  # PyMuPDF

def _fig_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def _drawdown(cum):
    roll_max = cum.cummax()
    return cum / roll_max - 1.0

def build_pdf(port: pd.Series, panel: pd.DataFrame, weights: pd.Series, meta: dict) -> bytes:
    port = port.dropna()
    weights = weights.dropna()
    fig1, ax1 = plt.subplots(figsize=(7, 3))
    (1 + port).cumprod().plot(ax=ax1, lw=2, color="black")
    ax1.set_title("Cumulative Growth of £1")
    ax1.set_ylabel("Value"); ax1.grid(True, alpha=0.3)
    img1 = _fig_png_bytes(fig1)
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    dd = _drawdown((1 + port).cumprod())
    dd.plot(ax=ax2, color="crimson", lw=2)
    ax2.set_title("Drawdown"); ax2.set_ylabel("Drawdown"); ax2.grid(True, alpha=0.3)
    img2 = _fig_png_bytes(fig2)
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    colors = np.where(port >= 0, "#2ca02c", "#d62728")
    port.plot(kind="bar", ax=ax3, color=colors)
    ax3.set_title("Monthly Returns"); ax3.set_ylabel("Return"); ax3.grid(True, axis="y", alpha=0.3)
    img3 = _fig_png_bytes(fig3)
    doc = fitz.open()
    page = doc.new_page()
    summary = (
        f"Portfolio Report\n\n"
        f"Period: {port.index.min().date()} → {port.index.max().date()}\n"
        f"Managers: {', '.join(panel.columns)}\n"
        f"Weights: " + ", ".join(f"{k}: {float(v):.1%}" for k, v in weights.items()) + "\n"
        f"Settings: {meta}\n"
    )
    page.insert_textbox(fitz.Rect(36, 36, 559, 200), summary, fontsize=12, fontname="helv")
    for img in (img1, img2, img3):
        page = doc.new_page()
        page.insert_image(fitz.Rect(36, 36, 559, 400), stream=img)
    return doc.tobytes()
