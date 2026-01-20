# app.py
import re
import numpy as np
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

CSV_PATH = "car_price_prediction.csv"

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Car Price Studio",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# CUSTOM CSS (nueva paleta)
# ---------------------------
CUSTOM_CSS = """
<style>
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(circle at 15% 15%, rgba(255,183,77,0.12), transparent 45%),
    radial-gradient(circle at 85% 20%, rgba(38,198,218,0.10), transparent 40%),
    radial-gradient(circle at 20% 85%, rgba(255,138,101,0.10), transparent 45%),
    #0f1117;
}
[data-testid="stHeader"]{background: rgba(0,0,0,0);}

:root{
  --card: rgba(255,255,255,0.07);
  --card-border: rgba(255,255,255,0.12);
  --text: rgba(255,255,255,0.94);
  --muted: rgba(255,255,255,0.70);
  --accent: rgba(255,183,77,0.95);
  --accent-2: rgba(38,198,218,0.90);
}

h1,h2,h3,h4,h5,h6, p, label, div, span { color: var(--text); }
small { color: var(--muted); }

.block-container { padding-top: 1.4rem; }

div[data-testid="stMetric"]{
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 18px;
  padding: 14px;
  box-shadow: 0 12px 34px rgba(0,0,0,0.35);
}

button[data-baseweb="tab"]{
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 14px !important;
}
button[data-baseweb="tab"][aria-selected="true"]{
  border: 1px solid rgba(255,183,77,0.85) !important;
  box-shadow: 0 0 0 2px rgba(255,183,77,0.25);
}

section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.04);
  border-right: 1px solid rgba(255,255,255,0.10);
}

button[kind="primary"]{
  background: linear-gradient(135deg, #ffb74d, #ff8a65) !important;
  color: #1a1a1a !important;
  border-radius: 14px !important;
  font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------
def parse_mileage(series):
    s = series.astype(str).str.replace(" km", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def engine_turbo_flag(series):
    return series.astype(str).str.contains("Turbo", case=False, na=False).astype(int)

def engine_numeric(series):
    s = series.astype(str).str.replace(" Turbo", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def engine_bucket(x):
    if pd.isna(x):
        return "Unknown"
    if x >= 4.5:
        return "4.5+"
    lo = np.floor(x / 0.5) * 0.5
    hi = lo + 0.5
    return f"{lo:.1f}-{hi:.1f}"

def clean_dataframe(raw):
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    if "Mileage" in df.columns:
        df["Mileage"] = parse_mileage(df["Mileage"])

    if "Engine volume" in df.columns:
        df["Is_Turbo"] = engine_turbo_flag(df["Engine volume"])
        df["_eng"] = engine_numeric(df["Engine volume"])
        df["Engine_volume_category"] = df["_eng"].apply(engine_bucket)
        df.drop(columns=["Engine volume", "_eng"], inplace=True)

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    return df

# ---------------------------
# Model
# ---------------------------
def make_pipeline(cfg):
    xgb = XGBRegressor(
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
        **cfg
    )
    return xgb

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns

    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )
    return pre

# ---------------------------
# Header
# ---------------------------
st.markdown("## 🚗 Car Price Studio")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.markdown("### ⚙️ Configuración")

uploaded = CSV_PATH if os.path.exists(CSV_PATH) else st.sidebar.file_uploader("Sube CSV", type=["csv"])

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
cfg = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
)

# ---------------------------
# Load data
# ---------------------------
if uploaded is None:
    st.info("Sube un CSV para comenzar")
    st.stop()

df = clean_dataframe(pd.read_csv(uploaded))

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧠 Entrenar", "🔮 Predicción"])

with tab1:
    st.metric("Filas", df.shape[0])
    st.metric("Columnas", df.shape[1])
    st.dataframe(df.head(20), use_container_width=True)

    if "Price" in df.columns:
        fig = px.histogram(
            df, x="Price",
            template="plotly_dark",
            color_discrete_sequence=["#ffb74d"]
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if "Price" not in df.columns:
        st.error("Falta la columna Price")
        st.stop()

    if st.button("🚀 Entrenar modelo", type="primary"):
        X = df.drop(columns="Price")
        y = df["Price"]

        pre = build_preprocessor(X)
        model = make_pipeline(cfg)

        pipe = Pipeline([("pre", pre), ("model", model)])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)

        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)

        st.metric("R²", f"{r2_score(yte, preds):.4f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(yte, preds)):.2f}")
        st.metric("MAE", f"{mean_absolute_error(yte, preds):.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yte, y=preds, mode="markers",
            marker=dict(color="rgba(255,183,77,0.85)", size=7, opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=[yte.min(), yte.max()],
            y=[yte.min(), yte.max()],
            mode="lines",
            line=dict(color="rgba(38,198,218,0.9)", dash="dash")
        ))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.session_state["pipe"] = pipe

with tab3:
    if "pipe" not in st.session_state:
        st.warning("Entrena el modelo primero")
        st.stop()

    pipe = st.session_state["pipe"]
    input_data = {}

    for col in df.drop(columns="Price").columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            input_data[col] = st.number_input(col, float(df[col].median()))
        else:
            input_data[col] = st.selectbox(col, df[col].unique())

    if st.button("🔮 Predecir", type="primary"):
        one = pd.DataFrame([input_data])
        pred = pipe.predict(one)[0]
        st.markdown(
            f"<div style='background:rgba(255,183,77,0.18);padding:18px;border-radius:18px;'>"
            f"💰 <b>Precio estimado:</b> <span style='font-size:1.6rem;'>${pred:,.0f}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
