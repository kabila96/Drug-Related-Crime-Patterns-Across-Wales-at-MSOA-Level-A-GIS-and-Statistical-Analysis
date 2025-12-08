
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import folium
from folium.plugins import HeatMap, FastMarkerCluster
from streamlit_folium import st_folium

# -----------------------------------------------------
# Page configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="Spatial Inequalities & Drug-Related Crime - Wales",
    layout="wide"
)

# -----------------------------------------------------
# Title and header
# -----------------------------------------------------
st.title("Spatial Inequalities, Socioeconomic Deprivation, and Drug-Related Crime Patterns Across Wales at MSOA Level: A GIS and Statistical Analysis.")
st.markdown("""
**By Powell Ndlovu**  
UNIVERSITY OF ABERTAWE / SWANSEA – **Chevening Scholar, MSc GIS and Climate Change**  
Data source: `wales.csv` (MSOA-level data for Wales)
""")

st.markdown("---")

# -----------------------------------------------------
# Data loading
# -----------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Fallback to local CSV name expected in same folder as app.py
        df = pd.read_csv("wales.csv")
    return df

st.sidebar.header("Data Options")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file (optional, must match wales.csv structure)", 
    type=["csv"]
)

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.sidebar.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# -----------------------------------------------------
# Basic data preview
# -----------------------------------------------------
st.subheader("1. Data Preview")
st.dataframe(df.head())

# -----------------------------------------------------
# Missingness analysis
# -----------------------------------------------------
st.subheader("2. Missing Data Analysis")

missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df)) * 100
missing_table = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage (%)": missing_pct.round(2)
})
st.dataframe(missing_table)

# -----------------------------------------------------
# Summary statistics
# -----------------------------------------------------
st.subheader("3. Summary Statistics (Numerical Variables)")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    st.dataframe(df[numeric_cols].describe().T)
else:
    st.info("No numeric columns detected.")

# -----------------------------------------------------
# Correlation matrix & heatmap
# -----------------------------------------------------
st.subheader("4. Correlation Matrix")

key_vars_default = ["DrugCrimes", "Total_Population", "Employment_DS", "Income_DS"]
key_vars = [c for c in key_vars_default if c in df.columns]

selected_corr_cols = st.multiselect(
    "Select variables for correlation matrix:",
    options=numeric_cols,
    default=key_vars if key_vars else numeric_cols
)

if len(selected_corr_cols) >= 2:
    corr = df[selected_corr_cols].corr()
    st.write("Correlation table")
    st.dataframe(corr)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
else:
    st.info("Select at least two numeric variables for correlation.")

# -----------------------------------------------------
# Top MSOA by DrugCrimes
# -----------------------------------------------------
st.subheader("5. Top MSOA Areas by Drug-Related Crime")

msoa_col_candidates = [c for c in df.columns if "MSOA" in c.upper() or "MSOA01" in c.upper()]
if msoa_col_candidates and "DrugCrimes" in df.columns:
    msoa_col = msoa_col_candidates[0]

    top_n = st.slider("Select number of top MSOA areas to display", 5, 30, 20)
    top_df = df.nlargest(top_n, "DrugCrimes").dropna(subset=[msoa_col])

    st.dataframe(top_df[[msoa_col, "DrugCrimes"]])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_df[msoa_col].astype(str), top_df["DrugCrimes"])
    ax.set_xlabel("Drug Crimes")
    ax.set_ylabel("MSOA")
    ax.set_title(f"Top {top_n} MSOA Areas by Drug-Related Crime")
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Columns 'DrugCrimes' and an MSOA name column are required for this section.")

# -----------------------------------------------------
# Regression analysis
# -----------------------------------------------------
st.subheader("6. Regression Analysis: Predicting Drug Crimes")

if "DrugCrimes" in df.columns:
    # Select predictors from numeric columns
    potential_predictors = [c for c in numeric_cols if c != "DrugCrimes"]
    predictors = st.multiselect(
        "Select predictor variables for regression model (Y = DrugCrimes):",
        options=potential_predictors,
        default=[c for c in ["Total_Population", "Employment_DS"] if c in potential_predictors]
    )

    if len(predictors) >= 1:
        reg_df = df[["DrugCrimes"] + predictors].dropna()

        X = reg_df[predictors]
        y = reg_df["DrugCrimes"]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        st.write("### Regression Summary")
        st.text(model.summary())

        # Show tidy coefficient table
        coef_table = pd.DataFrame({
            "Coefficient": model.params,
            "Std. Error": model.bse,
            "t-value": model.tvalues,
            "p-value": model.pvalues
        })
        st.dataframe(coef_table.round(4))
    else:
        st.info("Select at least one predictor variable for regression.")
else:
    st.info("Column 'DrugCrimes' is required for regression analysis.")

# -----------------------------------------------------
# Scatterplots with regression lines
# -----------------------------------------------------
st.subheader("7. Scatterplots with Regression Lines")

if "DrugCrimes" in df.columns and numeric_cols:
    x_var = st.selectbox(
        "Select an X-variable for scatterplot vs DrugCrimes:",
        options=[c for c in numeric_cols if c != "DrugCrimes"],
        index=0
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(x=df[x_var], y=df["DrugCrimes"], scatter_kws={"alpha": 0.5}, ax=ax)
    ax.set_xlabel(x_var)
    ax.set_ylabel("DrugCrimes")
    ax.set_title(f"DrugCrimes vs {x_var}")
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Numeric columns and 'DrugCrimes' required for scatterplots.")

# -----------------------------------------------------
# Geographic Heat Map (Folium)
# -----------------------------------------------------
st.subheader("8. Geographic Heat Map of Drug-Related Crime")

# Try to infer latitude, longitude and value columns from typical names
lat_candidates = [c for c in df.columns if c.upper() in ["LAT", "LATITUDE"]]
lon_candidates = [c for c in df.columns if c.upper() in ["LONG", "LON", "LNG", "LONGITUDE"]]
value_col = None
if "Number_DrugAbuse" in df.columns:
    value_col = "Number_DrugAbuse"
elif "DrugCrimes" in df.columns:
    value_col = "DrugCrimes"

if lat_candidates and lon_candidates and value_col is not None:
    lat_col = lat_candidates[0]
    lon_col = lon_candidates[0]

    map_df = df[[lat_col, lon_col, value_col]].dropna()

    if not map_df.empty:
        # Base map centred roughly on Wales (or mean of points)
        center_lat = map_df[lat_col].mean()
        center_lon = map_df[lon_col].mean()
        base_map = folium.Map(location=[center_lat, center_lon], zoom_start=7)

        heat_data = map_df[[lat_col, lon_col, value_col]].values.tolist()
        HeatMap(heat_data, radius=10, blur=15).add_to(base_map)

        st_folium(base_map, width=800, height=500)
    else:
        st.info("No valid rows with LAT/LONG and crime values to plot.")
else:
    st.info("Columns for latitude (e.g. 'LAT') and longitude (e.g. 'LONG') and a crime value column ('Number_DrugAbuse' or 'DrugCrimes') are required for the heat map.")

# -----------------------------------------------------
# Cluster Map (FastMarkerCluster)
# -----------------------------------------------------
st.subheader("9. Cluster Map of Drug-Related Crime Points")

if lat_candidates and lon_candidates and value_col is not None:
    lat_col = lat_candidates[0]
    lon_col = lon_candidates[0]

    map_df = df[[lat_col, lon_col, value_col]].dropna()

    if not map_df.empty:
        center_lat = map_df[lat_col].mean()
        center_lon = map_df[lon_col].mean()
        cluster_map = folium.Map(location=[center_lat, center_lon], zoom_start=7)

        # FastMarkerCluster expects a list of [lat, lon, ...]
        FastMarkerCluster(map_df[[lat_col, lon_col, value_col]].values.tolist()).add_to(cluster_map)

        st_folium(cluster_map, width=800, height=500)
    else:
        st.info("No valid rows with LAT/LONG and crime values to plot for clustering.")
else:
    st.info("Columns for latitude, longitude, and crime values are required for the cluster map.")

# -----------------------------------------------------
# Downloadable processed data (optional)
# -----------------------------------------------------
st.subheader("10. Download Processed Data")

csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download current dataset as CSV",
    data=csv_data,
    file_name="wales_processed.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("Built as part of a GIS & Crime Analysis workflow for Wales at MSOA level.")
