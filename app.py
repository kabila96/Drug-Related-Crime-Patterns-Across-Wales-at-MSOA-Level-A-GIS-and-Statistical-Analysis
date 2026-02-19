
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import statsmodels.api as sm
import folium
from folium.plugins import HeatMap, FastMarkerCluster
from streamlit_folium import st_folium

from pathlib import Path
from io import BytesIO

# Optional PDF export (executive summary)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Spatial Inequalities & Drug-Related Crime (Wales, MSOA)",
    layout="wide"
)

# -----------------------------
# Helpers
# -----------------------------
FIG_NO = 0
def fig_caption(text: str):
    """Write a numbered caption under a figure."""
    global FIG_NO
    FIG_NO += 1
    st.markdown(f"<div class='figcap'><b>Figure {FIG_NO}.</b> {text}</div>", unsafe_allow_html=True)

def safe_div(a, b):
    b = np.where(np.asarray(b)==0, np.nan, b)
    return np.asarray(a) / b

def qcut5(s):
    # robust quintiles (handles duplicates)
    try:
        return pd.qcut(s, 5, labels=["Q1 (least)","Q2","Q3","Q4","Q5 (most)"])
    except Exception:
        return pd.cut(s.rank(method="average"), 5, labels=["Q1 (least)","Q2","Q3","Q4","Q5 (most)"])

def make_exec_pdf(summary_dict, top_table, corr_table, model_summary_text):
    """
    Create a compact executive PDF (1–2 pages) using ReportLab.
    If ReportLab is missing, this function should never be called.
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    # Header
    c.setFont("Times-Bold", 16)
    c.drawString(2*cm, h-2.2*cm, "Executive Summary: Spatial Inequalities & Drug-Related Crime (Wales, MSOA)")
    c.setFont("Times-Roman", 11)
    c.drawString(2*cm, h-3.0*cm, "GIS Report by Powell A. Ndlovu | Chevening Scholar | MSc GIS & Climate Change (Swansea University)")

    # Key metrics
    y = h-4.2*cm
    c.setFont("Times-Bold", 12)
    c.drawString(2*cm, y, "Key metrics (from the filtered dataset)")
    y -= 0.6*cm
    c.setFont("Times-Roman", 11)

    for k, v in summary_dict.items():
        c.drawString(2*cm, y, f"• {k}: {v}")
        y -= 0.5*cm
        if y < 4*cm:
            c.showPage()
            y = h-2.5*cm

    # Top hotspots table (top 10)
    y -= 0.2*cm
    c.setFont("Times-Bold", 12)
    c.drawString(2*cm, y, "Top 10 MSOAs by drug-crime rate (per 1,000 people)")
    y -= 0.7*cm
    c.setFont("Times-Roman", 9)

    cols = ["MSOA01NAME", "DrugCrimes", "Total_Population", "Crimes_per_1000", "Income_Q", "Employment_Q"]
    tt = top_table[cols].copy()
    tt["Crimes_per_1000"] = tt["Crimes_per_1000"].round(2)

    # print simple table
    header = ["MSOA", "Crimes", "Pop", "Rate/1k", "Income", "Employ"]
    x = [2*cm, 9.5*cm, 11.2*cm, 12.8*cm, 14.6*cm, 17.0*cm]
    c.setFont("Times-Bold", 9)
    for xi, lab in zip(x, header):
        c.drawString(xi, y, lab)
    y -= 0.35*cm
    c.setFont("Times-Roman", 9)
    for _, r in tt.head(10).iterrows():
        c.drawString(x[0], y, str(r["MSOA01NAME"])[:40])
        c.drawRightString(x[1]+1.2*cm, y, f"{r['DrugCrimes']:.0f}")
        c.drawRightString(x[2]+1.0*cm, y, f"{r['Total_Population']:.0f}")
        c.drawRightString(x[3]+1.0*cm, y, f"{r['Crimes_per_1000']:.2f}")
        c.drawString(x[4], y, str(r["Income_Q"]))
        c.drawString(x[5], y, str(r["Employment_Q"]))
        y -= 0.32*cm
        if y < 3.2*cm:
            c.showPage()
            y = h-2.5*cm

    # Correlations
    y -= 0.3*cm
    c.setFont("Times-Bold", 12)
    c.drawString(2*cm, y, "Associations (Pearson correlations)")
    y -= 0.6*cm
    c.setFont("Times-Roman", 10)
    for line in corr_table:
        c.drawString(2*cm, y, f"• {line}")
        y -= 0.45*cm
        if y < 3.2*cm:
            c.showPage()
            y = h-2.5*cm

    # Model note
    y -= 0.2*cm
    c.setFont("Times-Bold", 12)
    c.drawString(2*cm, y, "Model note (OLS; exploratory)")
    y -= 0.6*cm
    c.setFont("Times-Roman", 9)
    for chunk in model_summary_text.split("\n"):
        c.drawString(2*cm, y, chunk[:110])
        y -= 0.35*cm
        if y < 3.0*cm:
            c.showPage()
            y = h-2.5*cm

    # References
    y -= 0.2*cm
    c.setFont("Times-Bold", 12)
    c.drawString(2*cm, y, "Key references (as cited in the full report)")
    y -= 0.55*cm
    c.setFont("Times-Roman", 9)
    refs = [
        "Ndlovu, P.A. (2026). Mini GIS Report: Spatial Inequalities, Socioeconomic Deprivation and Drug-Related Crime Patterns across Wales at MSOA level.",
        "Fischer, B., et al. (2010). Drugs: Education, Prevention and Policy, 17, 333–353.",
        "Sundquist, K., & Frank, G. (2004). Addiction, 99, 1298–1305."
    ]
    for r in refs:
        c.drawString(2*cm, y, f"• {r[:120]}")
        y -= 0.4*cm
        if y < 2.6*cm:
            c.showPage()
            y = h-2.5*cm

    c.save()
    buf.seek(0)
    return buf.getvalue()


# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
.figcap {
  margin-top: -0.5rem;
  margin-bottom: 1.0rem;
  font-size: 0.95rem;
  opacity: 0.85;
}
.smallnote {font-size: 0.92rem; opacity: 0.85;}
.kpi {padding: 0.9rem; border: 1px solid rgba(0,0,0,0.08); border-radius: 14px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.title("Spatial Inequalities, Socioeconomic Deprivation & Drug-Related Crime (Wales, MSOA)")
st.markdown("""
**GIS Report by Powell A. Ndlovu**  
Chevening Scholar | MSc GIS & Climate Change | Swansea University  
""")
st.markdown("---")

# -----------------------------
# Data load
# -----------------------------
BASE = Path(__file__).parent
data_path = BASE / "wales.csv"
report_path = BASE / "Mini_GIS_Report_Powell_Ndlovu.pdf"  # cleaned cover line

df = pd.read_csv(data_path)
df.columns = [c.strip() for c in df.columns]

required = ["MSOA01NAME","DrugCrimes","Total_Population","Employment_DS","Income_DS","LONG","LAT"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Dataset is missing required columns: {missing}")
    st.stop()

# Derived metrics
df["Crimes_per_1000"] = safe_div(df["DrugCrimes"], df["Total_Population"]) * 1000
df["Income_Q"] = qcut5(df["Income_DS"])
df["Employment_Q"] = qcut5(df["Employment_DS"])

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Filters")
msoa_search = st.sidebar.text_input("Search MSOA name", value="")
income_range = st.sidebar.slider(
    "Income deprivation score range",
    float(df["Income_DS"].min()), float(df["Income_DS"].max()),
    (float(df["Income_DS"].quantile(0.05)), float(df["Income_DS"].quantile(0.95)))
)
employ_range = st.sidebar.slider(
    "Employment deprivation score range",
    float(df["Employment_DS"].min()), float(df["Employment_DS"].max()),
    (float(df["Employment_DS"].quantile(0.05)), float(df["Employment_DS"].quantile(0.95)))
)
show_top_n = st.sidebar.slider("Show Top-N hotspots", 5, 30, 10)

# Filter
dff = df.copy()
if msoa_search.strip():
    dff = dff[dff["MSOA01NAME"].str.contains(msoa_search.strip(), case=False, na=False)]
dff = dff[(dff["Income_DS"]>=income_range[0]) & (dff["Income_DS"]<=income_range[1])]
dff = dff[(dff["Employment_DS"]>=employ_range[0]) & (dff["Employment_DS"]<=employ_range[1])]

if len(dff) == 0:
    st.warning("No rows match the current filters.")
    st.stop()

# -----------------------------
# Context / Introduction
# -----------------------------
with st.expander("Study context (short)", expanded=True):
    st.markdown("""
This analysis explores how **drug-related crime counts** vary across Wales at **MSOA scale**, and how observed spatial patterns relate to
two area-level socioeconomic indicators: **income deprivation** and **employment deprivation**. The purpose is descriptive and exploratory:
to surface spatial inequalities and provide evidence that can support targeted, place-based policy discussions.

The framing aligns with established work linking community disadvantage and substance-related harms at aggregate levels, while recognising that
area-level relationships do not, by themselves, establish individual-level causality (**ecological inference caution**).
""")
    st.markdown("<div class='smallnote'><b>Source:</b> Ndlovu (2026) – Mini GIS Report (full report available for download in this app).</div>", unsafe_allow_html=True)

# -----------------------------
# Executive snapshot
# -----------------------------
st.subheader("Executive snapshot (from filtered data)")
col1, col2, col3, col4 = st.columns(4)

total_crimes = float(dff["DrugCrimes"].sum())
total_pop = float(dff["Total_Population"].sum())
mean_rate = float(dff["Crimes_per_1000"].mean())
median_rate = float(dff["Crimes_per_1000"].median())

col1.metric("MSOAs (n)", f"{len(dff):,}")
col2.metric("Drug crimes (sum)", f"{total_crimes:,.0f}")
col3.metric("Mean rate / 1,000", f"{mean_rate:.2f}")
col4.metric("Median rate / 1,000", f"{median_rate:.2f}")

st.markdown("---")

# -----------------------------
# Interactive visuals
# -----------------------------
left, right = st.columns([1.05, 0.95])

with left:
    st.subheader("Spatial distribution (interactive map)")
    # folium map
    center = [float(dff["LAT"].mean()), float(dff["LONG"].mean())]
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    # heatmap weights: crime rate
    heat = dff[["LAT","LONG","Crimes_per_1000"]].dropna().values.tolist()
    HeatMap(heat, radius=18, blur=22, max_zoom=9).add_to(m)

    # marker cluster with popups
    def popup_html(r):
        return f"""
        <b>{r['MSOA01NAME']}</b><br>
        Crimes: {r['DrugCrimes']:.0f}<br>
        Population: {r['Total_Population']:.0f}<br>
        Rate/1,000: {r['Crimes_per_1000']:.2f}<br>
        Income DS: {r['Income_DS']:.3f} ({r['Income_Q']})<br>
        Employment DS: {r['Employment_DS']:.3f} ({r['Employment_Q']})
        """
    cluster = FastMarkerCluster(
        data=dff[["LAT","LONG"]].dropna().values.tolist()
    )
    cluster.add_to(m)

    # Add circle markers for top N hotspots for visibility
    top = dff.sort_values("Crimes_per_1000", ascending=False).head(show_top_n)
    for _, r in top.iterrows():
        folium.CircleMarker(
            location=[r["LAT"], r["LONG"]],
            radius=8,
            popup=folium.Popup(popup_html(r), max_width=320),
            tooltip=f"{r['MSOA01NAME']} | {r['Crimes_per_1000']:.2f} /1k",
            color="#111111",
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    st_folium(m, height=520, width=None)
    fig_caption("Heat intensity reflects drug-crime rate per 1,000 population. Circle markers highlight the Top‑N MSOAs under the current filters.")

with right:
    st.subheader("Hotspots table (Top‑N)")
    top_tbl = dff.sort_values("Crimes_per_1000", ascending=False).head(show_top_n).copy()
    show_cols = ["MSOA01NAME","DrugCrimes","Total_Population","Crimes_per_1000","Income_DS","Employment_DS","Income_Q","Employment_Q"]
    top_tbl["Crimes_per_1000"] = top_tbl["Crimes_per_1000"].round(2)
    st.dataframe(top_tbl[show_cols], use_container_width=True, height=520)
    fig_caption("Top MSOAs are ranked by drug-crime rate (per 1,000 people), helping prioritise places for closer investigation.")

st.markdown("---")

# Chart row 1
c1, c2 = st.columns(2)

with c1:
    st.subheader("Distribution of drug-crime rate")
    fig = px.histogram(
        dff, x="Crimes_per_1000", nbins=30,
        hover_data=["MSOA01NAME","DrugCrimes","Total_Population"]
    )
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)
    fig_caption("Most MSOAs cluster around lower rates, with a smaller number of high-rate MSOAs forming the upper tail.")

with c2:
    st.subheader("Rates by deprivation quintiles (Income)")
    fig = px.violin(
        dff, x="Income_Q", y="Crimes_per_1000", box=True, points="all",
        hover_data=["MSOA01NAME"]
    )
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)
    fig_caption("Comparing distributions by quintile helps assess whether higher deprivation areas tend to have higher drug-crime rates (descriptive).")

# Chart row 2
c3, c4 = st.columns(2)

with c3:
    st.subheader("Income deprivation vs crime rate")
    fig = px.scatter(
        dff, x="Income_DS", y="Crimes_per_1000",
        size="Total_Population", hover_name="MSOA01NAME",
        trendline="ols"
    )
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=400)
    st.plotly_chart(fig, use_container_width=True)
    fig_caption("Scatter with OLS trendline summarises the direction of association at MSOA level (not causal). Bubble size reflects population.")

with c4:
    st.subheader("Employment deprivation vs crime rate")
    fig = px.scatter(
        dff, x="Employment_DS", y="Crimes_per_1000",
        size="Total_Population", hover_name="MSOA01NAME",
        trendline="ols"
    )
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=400)
    st.plotly_chart(fig, use_container_width=True)
    fig_caption("This plot provides an aligned view for employment deprivation. Interactivity allows inspection of specific MSOAs driving the pattern.")

st.markdown("---")

# -----------------------------
# Statistics (transparent + cautious)
# -----------------------------
st.subheader("Statistical summary (exploratory)")

# Correlations
corr_vars = dff[["Crimes_per_1000","Income_DS","Employment_DS","Total_Population"]].dropna()
corr = corr_vars.corr(method="pearson")

fig = px.imshow(
    corr, text_auto=".2f", aspect="auto",
)
fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=360)
st.plotly_chart(fig, use_container_width=True)
fig_caption("Correlation matrix summarises linear associations among rate, deprivation indicators, and population (within filtered data).")

# OLS model (rate ~ income + employment + log(pop))
d = corr_vars.copy()
d["log_pop"] = np.log(d["Total_Population"].clip(lower=1))
X = d[["Income_DS","Employment_DS","log_pop"]]
X = sm.add_constant(X)
y = d["Crimes_per_1000"]
model = sm.OLS(y, X).fit()

with st.expander("OLS output (rate per 1,000 ~ income + employment + log(pop))", expanded=False):
    st.text(model.summary().as_text())

st.markdown("""
<div class="smallnote">
<b>Interpretation note:</b> The model is presented as an exploratory summary of associations at MSOA level.
It does not prove causation and may be influenced by unmeasured factors (e.g., policing intensity, land-use context, service access).
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Key findings & discussion (text generated from actual filtered data)
# -----------------------------
st.subheader("Key findings and discussion (grounded in the displayed data)")

top1 = dff.sort_values("Crimes_per_1000", ascending=False).head(1).iloc[0]
corr_income = float(corr.loc["Crimes_per_1000","Income_DS"])
corr_employ = float(corr.loc["Crimes_per_1000","Employment_DS"])

st.markdown(f"""
**Finding 1 — Concentration of high rates in a small set of MSOAs.**  
Under the current filters, the highest observed drug-crime rate is **{top1['Crimes_per_1000']:.2f} per 1,000** in **{top1['MSOA01NAME']}**.

**Finding 2 — Deprivation indicators show measurable associations with rates (descriptive).**  
Pearson correlation with drug-crime rate: **Income_DS r = {corr_income:.2f}**, **Employment_DS r = {corr_employ:.2f}** (filtered data).

**Finding 3 — Spatial clustering is visible on the map layer.**  
The heat layer and Top‑N markers indicate that some neighbouring MSOAs share elevated rates, suggesting geographically localised patterns worth follow‑up.
""")

st.markdown("""
**Practical discussion points (within study scope):**
- These outputs help prioritise **where** to investigate further and **which MSOAs** to compare, rather than asserting why outcomes occur.
- For applied use, consider complementing this dataset with: temporal crime series, service access indicators, policing/resource allocation measures, and land‑use context, while maintaining careful causal language.

**Possible policy/monitoring directions (evidence-informed, non-causal):**
- Targeted, place-based prevention and treatment outreach in repeated hotspot MSOAs.
- Cross-sector collaboration (public health + local authorities) to align deprivation alleviation strategies with harm reduction.
- Routine dashboard updates as new data arrives (monitoring change rather than one-off description).
""")

st.markdown("---")

# -----------------------------
# Downloads
# -----------------------------
st.subheader("Downloads")

dl1, dl2 = st.columns([1,1])

with dl1:
    st.markdown("**Full report (PDF)**")
    if report_path.exists():
        with open(report_path, "rb") as f:
            st.download_button(
                label="Download full report (PDF)",
                data=f.read(),
                file_name=report_path.name,
                mime="application/pdf",
                use_container_width=True
            )
        st.caption("This is the attached report with the cover line updated to: “GIS REPORT BY POWELL A NDLOVU”.")
    else:
        st.warning("Full report PDF not found next to the app file. Expected: Mini_GIS_Report_Powell_Ndlovu.pdf")

with dl2:
    st.markdown("**Executive summary (PDF)**")
    if REPORTLAB_OK:
        summary = {
            "MSOAs included": f"{len(dff):,}",
            "Total drug crimes": f"{total_crimes:,.0f}",
            "Total population": f"{total_pop:,.0f}",
            "Mean rate per 1,000": f"{mean_rate:.2f}",
            "Median rate per 1,000": f"{median_rate:.2f}",
        }
        corr_lines = [
            f"Income deprivation vs rate: r = {corr_income:.2f}",
            f"Employment deprivation vs rate: r = {corr_employ:.2f}",
            f"Population vs rate: r = {float(corr.loc['Crimes_per_1000','Total_Population']):.2f}",
        ]
        model_text = "\n".join(model.summary().as_text().splitlines()[:18])  # first lines only
        pdf_bytes = make_exec_pdf(summary, top_tbl, corr_lines, model_text)

        st.download_button(
            label="Download executive summary (PDF)",
            data=pdf_bytes,
            file_name="Executive_Summary_Wales_MSOA_DrugCrime_Powell_Ndlovu.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        st.caption("Auto-generated from the filtered results shown in this dashboard.")
    else:
        st.info("Executive summary export requires `reportlab`. Add it to requirements.txt to enable this button.")

st.markdown("---")

# -----------------------------
# References
# -----------------------------
with st.expander("References", expanded=False):
    st.markdown("""
- Ndlovu, P. A. (2026). *Mini GIS Report: Spatial Inequalities, Socioeconomic Deprivation and Drug-Related Crime Patterns across Wales at MSOA Level: A GIS and Statistical Analysis.* (Course report).
- Fischer, B., Rudzinski, K., Ivsins, A., Gallupe, O., Patra, J., & Krajden, M. (2010). Social, health and drug use characteristics of primary crack users in three mid-sized communities in British Columbia, Canada. *Drugs: Education, Prevention and Policy*, 17, 333–353.
- Sundquist, K., & Frank, G. (2004). Urbanization and hospital admission rates for alcohol and drug abuse: a follow-up study of 4.5 million women and men in Sweden. *Addiction*, 99, 1298–1305.
""")
