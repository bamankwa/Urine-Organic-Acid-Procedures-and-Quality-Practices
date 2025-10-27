import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Urine Organic Acids — Survey Dashboard", layout="wide")

DEFAULT_CSV = "Urine Organic Procedures.csv"

TABLE1_CATEGORIES = [
    "Years of performing OA testing",
    "Number of OA tested per year",
    "Type of instrument used for OA",
    "Vendors of instruments used",
    "Run time for routine analysis (minutes)",
]
TABLE2_CATEGORIES = [
    "Reasons for creatinine normalization",
    "Special procedures for high/low creatinine samples?",
    "Specimen rejection based on minimum creatinine results?",
    "Criteria used by lab to reject urine sample",
]
TABLE3_CATEGORIES = [
    "Chemicals used for extraction",
    "Does your laboratory perform oximation?",
    "Reagent Laboratory is using for oximation",
    "Derivatization Method",
]
TABLE4_CATEGORIES = [
    "Spikes IS before extraction to track efficiency",
    "Spikes ES post-extraction, pre-evaporation to verify instrument performance",
    "What is the number of internal standards used in your procedure?",
    "Why laboratories use more than one internal standard",
]
TABLE5_CATEGORIES = [
    "Type of internal standard for monitoring pre-analytical steps",
    "Type of internal standard for quantitation/semi-quantitation purposes",
]
TABLE6_CATEGORIES = [
    "Does your laboratory perform quantitative analysis",
    "How many compounds are quantified",
    "Number of standard mixes prepared for calibration",
    "Units of quantitation",
    "Are the concentrations provided on the final report",
]
TABLE7_CATEGORIES = [
    "Inclusion of QC materials in each run",
    "Number of QC materials with each run",
    "Materials used for QC",
    "Number of compounds monitored in QC samples",
    "External proficiency testing for organic acids",
]
TABLE8_CATEGORIES = [
    "Is lab performing an instrument or system suitability check",
    "Specimens used for instrument verification check",
    "Verification parameters before routine sample injection",
    "Number of analytes reviewed during instrument verification check",
]
TABLE9_CATEGORIES = [
    "Libraries used for peak identification",
    "How compounds are added to in-house library",
    "Overall process for peak identification",
    "Variables used for establishing peak identity",
]
TABLE10_CATEGORIES = [
    "Who interprets organic acid profiles",
    "Frequency of interpretation without access to clinical or phenotypic information",
    "Frequency of reports where lab seeks clinical info from providers/charts",
    "Do you correlate organic acids with other tests before interpretation",
    "What is included in the report?",
]

HEATMAP_CATEGORY_SINGLE = "Difficult-to-Analyze Analytes Across Participating Laboratories"
HEATMAP_CATEGORY_CHALLENGES = "Challenges Associated with the Top 10 Difficult-to-Analyze Analytes Across Participating Laboratories"
HEATMAP_CATEGORY_REMEDIES = "Remedies for the top 10 challenging analytes"

FIGURE_OPTIONS = {
    "Table 1 — General lab info": ("table", TABLE1_CATEGORIES),
    "Table 2 — Creatinine normalization & rejection": ("table", TABLE2_CATEGORIES),
    "Table 3 — Extraction / Oximation / Derivatization": ("table", TABLE3_CATEGORIES),
    "Table 4 — Internal/External Standards": ("table", TABLE4_CATEGORIES),
    "Table 5 — Standards for monitoring & quantification": ("table", TABLE5_CATEGORIES),
    "Table 6 — Quantification info": ("table", TABLE6_CATEGORIES),
    "Table 7 — Quality control practices": ("table", TABLE7_CATEGORIES),
    "Table 8 — Instrument performance / verification": ("table", TABLE8_CATEGORIES),
    "Table 9 — Peak identification & reporting": ("table", TABLE9_CATEGORIES),
    "Table 10 — Interpretation & reporting": ("table", TABLE10_CATEGORIES),
    "Heatmap — Difficult Analytes": ("heatmap_single", HEATMAP_CATEGORY_SINGLE),
    "Heatmap — Challenges × Top-10 Analytes": ("heatmap_matrix", HEATMAP_CATEGORY_CHALLENGES),
    "Heatmap — Remedies × Top-10 Analytes": ("heatmap_matrix", HEATMAP_CATEGORY_REMEDIES),
}

def load_csv(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if p.suffix.lower() != ".csv":
        p = p.with_suffix(".csv")
    return pd.read_csv(p)

def facet_barh(df: pd.DataFrame, categories: list[str], title: str, value: str = "percent", max_cols: int = 3):
    cat_type = pd.CategoricalDtype(categories, ordered=True)
    d = df[df["category"].isin(categories)].copy()
    if d.empty:
        st.warning(f"No data found for: {title}")
        return None
    d["category"] = d["category"].astype(cat_type)
    d = d.sort_values(by=["category", "order"] if "order" in d.columns else ["category", "response"]).copy()

    cats = list(d["category"].drop_duplicates())
    n = len(cats)
    ncols = min(max_cols, n) if n > 0 else 1
    nrows = int(np.ceil(max(n, 1) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4.2*nrows), squeeze=False)

    if value == "percent":
        xlim = (0, 100)
        x_label = "Percentage (%)"
    else:
        xlim = (0, max(d["numerator"].max(), 1) * 1.1)
        x_label = "Count (N)"

    for i, cat in enumerate(cats):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]

        sub = d[d["category"] == cat].copy()
        if "order" in sub.columns:
            sub = sub.sort_values("order")
        else:
            sub = sub.sort_values("response")

        if value == "percent":
            values = sub["percent"]
            labels = sub["response"]
        else:
            values = sub["numerator"]
            labels = sub["response"]

        ax.barh(labels, values)
        for y, (val, num, den, resp) in enumerate(zip(values, sub["numerator"], sub["denominator"], labels)):
            if value == "percent":
                text = f"{val:.1f}% ({num}/{int(den)})"
            else:
                pct = sub.loc[sub["response"]==resp, "percent"].values[0]
                text = f"{int(num)}/{int(den)} ({pct:.1f}%)"
            ax.text(val, y, "  " + text, va="center", ha="left")

        ax.set_title(str(cat), fontsize=12, pad=8)
        ax.set_xlim(xlim)
        ax.set_xlabel(x_label)
        ax.set_ylabel("")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    total_axes = nrows * ncols
    for j in range(n, total_axes):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

def heatmap_single_column(df: pd.DataFrame, category: str, title: str):
    sub = df[df["category"] == category].copy()
    if sub.empty:
        st.warning(f"No data found for: {title}")
        return None

    sub = sub.sort_values("order" if "order" in sub.columns else "response")
    labels = list(sub["response"])
    counts = sub["numerator"].astype(float).values.reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(7, 5 + 0.25 * len(labels)))
    im = ax.imshow(counts, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks([0])
    ax.set_xticklabels(["Labs_count"])
    for y in range(counts.shape[0]):
        ax.text(0, y, f"{int(counts[y,0])}", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.set_ylabel("Analyte")
    fig.tight_layout()
    return fig

def heatmap_matrix(df: pd.DataFrame, category: str, title: str, pivot_col: str):
    sub = df[df["category"] == category].copy()
    if sub.empty:
        st.warning(f"No data found for: {title}")
        return None

    # If explicit columns are not present, parse "response" like "Analyte | Thing"
    if pivot_col not in sub.columns:
        parts = sub["response"].str.split("|", n=1, expand=True)
        sub["analyte"] = parts[0].str.strip()
        sub[pivot_col] = parts[1].str.strip() if parts.shape[1] > 1 else ""
    sub["numerator"] = pd.to_numeric(sub["numerator"], errors="coerce").fillna(0)

    mat = sub.pivot_table(index="analyte", columns=pivot_col, values="numerator", aggfunc="sum", fill_value=0).sort_index()
    analytes = list(mat.index)
    cols = list(mat.columns)

    fig_h = 5 + 0.25 * len(analytes)
    fig_w = 6 + 0.35 * max(6, len(cols))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.values, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count")
    ax.set_yticks(np.arange(len(analytes)))
    ax.set_yticklabels(analytes)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right")
    ax.set_title(title)
    ax.set_xlabel(pivot_col.capitalize())
    ax.set_ylabel("Analyte")
    fig.tight_layout()
    return fig

st.title("Urine Organic Acids — Survey Dashboard")

with st.sidebar:
    st.header("Controls")
    csv_path = st.text_input("CSV path:", value=DEFAULT_CSV)
    figure_choice = st.selectbox("Figure:", list(FIGURE_OPTIONS.keys()))
    value_metric = st.radio("Metric (tables only):", ["percent", "numerator"], index=0)
    generate = st.button("Generate figure")

if generate:
    try:
        df = load_csv(csv_path)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.stop()

    kind, payload = FIGURE_OPTIONS[figure_choice]

    if kind == "table":
        # Determine sensible max_cols for selected table for layout
        title = figure_choice.split(" — ", 1)[-1]
        max_cols = 2 if "Table 2" in figure_choice or "Table 3" in figure_choice or "Table 4" in figure_choice or "Table 5" in figure_choice or "Table 6" in figure_choice or "Table 7" in figure_choice or "Table 8" in figure_choice or "Table 9" in figure_choice or "Table 10" in figure_choice else 3
        fig = facet_barh(df, payload, title, value=value_metric, max_cols=max_cols)
        if fig is not None:
            st.pyplot(fig)

    elif kind == "heatmap_single":
        fig = heatmap_single_column(df, payload, "Difficult-to-Analyze Analytes Across Participating Laboratories")
        if fig is not None:
            st.pyplot(fig)

    elif kind == "heatmap_matrix":
        pivot_col = "challenge" if payload == HEATMAP_CATEGORY_CHALLENGES else "remedy"
        pretty_title = "Challenges Associated with the Top 10 Difficult-to-Analyze Analytes Across Participating Laboratories" if payload == HEATMAP_CATEGORY_CHALLENGES else "Remedies for the top 10 challenging analytes"
        fig = heatmap_matrix(df, payload, pretty_title, pivot_col=pivot_col)
        if fig is not None:
            st.pyplot(fig)

    else:
        st.error("Unknown figure type.")
else:
    st.info("Select a figure and click **Generate figure** to render it.")

st.caption("Tip: run with `streamlit run urine_organic_dashboard.py`")
