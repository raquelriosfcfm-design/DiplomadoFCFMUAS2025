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
# Page config + CSS (bonito)
# ---------------------------
st.set_page_config(
    page_title="Car Price Studio",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Fondo elegante oscuro */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(circle at 15% 15%, rgba(255, 183, 77, 0.12), transparent 45%),
    radial-gradient(circle at 85% 20%, rgba(38, 198, 218, 0.10), transparent 40%),
    radial-gradient(circle at 20% 85%, rgba(255, 138, 101, 0.10), transparent 45%),
    #0f1117;
}
[data-testid="stHeader"]{background: rgba(0,0,0,0);}

:root{
  --card: rgba(255,255,255,0.07);
  --card-border: rgba(255,255,255,0.12);
  --text: rgba(255,255,255,0.94);
  --muted: rgba(255,255,255,0.70);
  --accent: rgba(255,183,77,0.95);      /* Ámbar */
  --accent-2: rgba(38,198,218,0.90);    /* Teal */
}

h1,h2,h3,h4,h5,h6, p, label, div, span { color: var(--text); }
small { color: var(--muted); }

/* Cards / métricas */
.block-container { padding-top: 1.4rem; }
div[data-testid="stMetric"]{
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 18px;
  padding: 14px 14px 10px 14px;
  box-shadow: 0 12px 34px rgba(0,0,0,0.35);
}

/* Tabs */
button[data-baseweb="tab"]{
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 14px !important;
  margin-right: 6px !important;
}
button[data-baseweb="tab"][aria-selected="true"]{
  border: 1px solid rgba(255,183,77,0.85) !important;
  box-shadow: 0 0 0 2px rgba(255,183,77,0.25);
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.04);
  border-right: 1px solid rgba(255,255,255,0.10);
}

/* Botones primarios */
button[kind="primary"]{
  background: linear-gradient(135deg, #ffb74d, #ff8a65) !important;
  color: #1a1a1a !important;
  border-radius: 14px !important;
  border: none !important;
  font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------
# Helpers: limpieza / features
# ---------------------------
def parse_mileage(series: pd.Series) -> pd.Series:
    # "186005 km" -> 186005
    s = series.astype(str).str.replace(" km", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def engine_turbo_flag(series: pd.Series) -> pd.Series:
    # "3.5 Turbo" -> 1
    s = series.astype(str)
    return s.str.contains("Turbo", case=False, na=False).astype(int)

def engine_numeric(series: pd.Series) -> pd.Series:
    # "3.5 Turbo" -> 3.5
    s = series.astype(str).str.replace(" Turbo", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def engine_bucket(x: float) -> str:
    if pd.isna(x):
        return "Unknown"
    if x >= 4.5:
        return "4.5+"
    lower = np.floor(x / 0.5) * 0.5
    upper = lower + 0.5
    return f"{lower:.1f}-{upper:.1f}"

def group_rare_categories(df: pd.DataFrame, col: str, quantile_threshold: float, other_label="Other") -> pd.DataFrame:
    freq = df[col].value_counts(dropna=False)
    thr = freq.quantile(quantile_threshold)
    rare = freq[freq < thr].index
    df[col] = df[col].where(~df[col].isin(rare), other_label)
    return df

def clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # Normaliza nombres por si vienen raros
    df.columns = [c.strip() for c in df.columns]

    # Drop ID si existe
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Levy a numeric
    if "Levy" in df.columns:
        df["Levy"] = pd.to_numeric(df["Levy"], errors="coerce")

    # Mileage a numeric
    if "Mileage" in df.columns:
        df["Mileage"] = parse_mileage(df["Mileage"])

    # Engine volume -> Is_Turbo + Engine_volume_category
    if "Engine volume" in df.columns:
        df["Is_Turbo"] = engine_turbo_flag(df["Engine volume"])
        df["_Engine_numeric"] = engine_numeric(df["Engine volume"])
        df["Engine_volume_category"] = df["_Engine_numeric"].apply(engine_bucket)
        df = df.drop(columns=["Engine volume", "_Engine_numeric"])

    # Strings clean
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    return df

# ---------------------------
# Model building
# ---------------------------
def make_pipeline(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0):
    # Separaremos cols dentro del fit según df
    xgb = XGBRegressor(
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    # ColumnTransformer se arma al vuelo luego
    return xgb

@st.cache_data(show_spinner=False)
def get_basic_profile(df: pd.DataFrame) -> dict:
    profile = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "nulls": int(df.isna().sum().sum()),
        "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
        "cat_cols": df.select_dtypes(exclude=[np.number]).columns.tolist(),
    }
    return profile

def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return pre, num_cols, cat_cols

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame, cfg: dict):
    # Separar target
    if "Price" not in df.columns:
        raise ValueError("No encuentro la columna 'Price' en el dataset.")

    X = df.drop(columns=["Price"])
    y = df["Price"].astype(float)

    pre, num_cols, cat_cols = build_preprocessor(X)
    model_cfg = {k: v for k, v in cfg.items() if k != "test_size"} 
    xgb = make_pipeline(**model_cfg)

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("model", xgb),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=42
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "y_test": y_test,
        "preds": preds,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "X_train": X_train,
    }

    return pipe, metrics

def plot_pred_vs_real(y_test, preds):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=preds, mode="markers",
        marker=dict(size=7, opacity=0.55),
        name="Predicciones"
    ))
    lo = float(min(y_test.min(), preds.min()))
    hi = float(max(y_test.max(), preds.max()))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(dash="dash"),
        name="Perfecto"
    ))
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Precio real",
        yaxis_title="Precio predicho",
    )
    return fig

def get_feature_names(pipe: Pipeline, X: pd.DataFrame):
    pre = pipe.named_steps["pre"]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # num names
    feature_names = []
    feature_names.extend(num_cols)

    # cat names (onehot)
    try:
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(ohe_names)
    except Exception:
        # fallback
        feature_names.extend([f"cat_{i}" for i in range(9999)])

    return feature_names

def plot_feature_importance(pipe: Pipeline, X_train: pd.DataFrame, top_k=25):
    model = pipe.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None

    names = get_feature_names(pipe, X_train)
    # recorta por seguridad
    n = min(len(names), len(importances))
    s = pd.Series(importances[:n], index=names[:n]).sort_values(ascending=False).head(top_k).reset_index()
    s.columns = ["Feature", "Importance"]

    fig = px.bar(
        s[::-1], x="Importance", y="Feature",
        orientation="h",
        template="plotly_dark",
        height=520,
        title=f"Top {top_k} variables más importantes (XGBoost)"
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig

# ---------------------------
# Header
# ---------------------------
colA, colB = st.columns([1.4, 1])
with colA:
    st.markdown("## 🚗 Car Price Studio")
with colB:
    st.markdown(
        "<div style='background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); "
        "border-radius: 18px; padding: 14px; box-shadow: 0 10px 30px rgba(0,0,0,0.25);'>"
        "<div style='font-size: 0.95rem; color: rgba(255,255,255,0.78);'>Tips</div>"
        "<ul style='margin-top: 8px; color: rgba(255,255,255,0.92);'>"
        "<li>Usa el <b>sidebar</b> para controlar rare categories y el modelo.</li>"
        "<li>En <b>Predicción</b>, puedes hacer una predicción individual o por lote.</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True
    )

st.write("")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.markdown("### ⚙️ Configuración")

uploaded = None  
if os.path.exists(CSV_PATH):     
    uploaded = CSV_PATH     
    st.sidebar.success("📁 Dataset cargado automáticamente desde el repositorio") 
else:     
    uploaded = st.sidebar.file_uploader("Sube tu CSV", type=["csv"])

st.sidebar.markdown("#### Sidebar 📊 EDA")
q_man = st.sidebar.slider("Manufacturer: quantile threshold", 0.10, 0.95, 0.65, 0.05)
q_model = st.sidebar.slider("Model: quantile threshold", 0.10, 0.99, 0.90, 0.01)
q_cat = st.sidebar.slider("Category: quantile threshold", 0.10, 0.95, 0.50, 0.05)
q_fuel = st.sidebar.slider("Fuel type: quantile threshold", 0.10, 0.95, 0.65, 0.05)
q_color = st.sidebar.slider("Color: quantile threshold", 0.10, 0.95, 0.75, 0.05)

st.sidebar.markdown("#### Modelo (XGBoost)")
test_size = st.sidebar.slider("Test size", 0.10, 0.40, 0.20, 0.05)
n_estimators = st.sidebar.slider("n_estimators", 200, 1400, 600, 50)
max_depth = st.sidebar.slider("max_depth", 2, 12, 6, 1)
learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.30, 0.05, 0.01)
subsample = st.sidebar.slider("subsample", 0.50, 1.00, 0.90, 0.05)
colsample_bytree = st.sidebar.slider("colsample_bytree", 0.50, 1.00, 0.90, 0.05)
reg_lambda = st.sidebar.slider("reg_lambda", 0.0, 5.0, 1.0, 0.25)

cfg = dict(
    test_size=test_size,
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    reg_lambda=reg_lambda,
)

# ---------------------------
# Load / clean data
# ---------------------------
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

if uploaded is None:
    st.info("📎 Sube tu CSV en el sidebar para empezar.")
    st.stop()

raw = load_csv(uploaded)
df = clean_dataframe(raw)

# Group rare categories like your notebook
for col, q in [
    ("Manufacturer", q_man),
    ("Model", q_model),
    ("Category", q_cat),
    ("Fuel type", q_fuel),
    ("Color", q_color),
]:
    if col in df.columns:
        df = group_rare_categories(df, col, q, other_label="Other")

profile = get_basic_profile(df)

# ---------------------------
# Tabs
# ---------------------------
tab_overview, tab_eda, tab_train, tab_predict = st.tabs(
    ["✨ Overview", "📊 EDA", "🧠 Entrenar", "🔮 Predicción"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Filas", f"{profile['rows']:,}")
    c2.metric("Columnas", f"{profile['cols']:,}")

    # Price promedio
    if "Price" in df.columns:
        c3.metric("Price promedio", f"${df['Price'].mean():,.0f}")
    else:
        c3.metric("Price promedio", "N/A")

    # Prod. year promedio
    # Ojo: tu dataset trae "Prod. year" (con punto y espacio)
    if "Prod. year" in df.columns:
        prod_year_mean = pd.to_numeric(df["Prod. year"], errors="coerce").mean()
        c4.metric("Prod. year promedio", f"{prod_year_mean:.0f}" if pd.notna(prod_year_mean) else "N/A")
    else:
        c4.metric("Prod. year promedio", "N/A")

    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("### Vista rápida del dataset")
        st.dataframe(df.head(30), use_container_width=True)

    with right:
        st.markdown("### Columnas")
        st.markdown(
            f"- **Numéricas**: {len(profile['numeric_cols'])}\n"
            f"- **Categóricas**: {len(profile['cat_cols'])}\n"
        )
        st.markdown("**Categóricas detectadas:**")
        st.write(profile["cat_cols"])

        if "Price" in df.columns:
            fig = px.histogram(df, x="Price", nbins=40, template="plotly_dark", title="Distribución de Price")
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

with tab_eda:
    st.markdown("### Explora variables")

    col1, col2 = st.columns([0.55, 0.45])
    with col1:
        target = "Price" if "Price" in df.columns else None
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        eda_mode = st.radio("Modo", ["Numérica", "Categórica"], horizontal=True)

        if eda_mode == "Numérica":
            col_sel = st.selectbox("Variable numérica", numeric_cols if numeric_cols else [])
            if col_sel:
                fig = px.histogram(df, x=col_sel, nbins=45, template="plotly_dark",
                                   title=f"Histograma: {col_sel}")
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            col_sel = st.selectbox("Variable categórica", cat_cols if cat_cols else [])
            topn = st.slider("Top N", 5, 30, 15)
            if col_sel:
                vc = df[col_sel].value_counts().head(topn).reset_index()
                vc.columns = [col_sel, "count"]
                fig = px.bar(vc, x=col_sel, y="count", template="plotly_dark",
                             title=f"Top {topn}: {col_sel}")
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Price" in df.columns:
            st.markdown("#### Relación con Price")
            x_num = st.selectbox("X (numérica)", [c for c in df.select_dtypes(include=[np.number]).columns if c != "Price"])
            if x_num:
                fig = px.scatter(df, x=x_num, y="Price", template="plotly_dark",
                                 title=f"{x_num} vs Price", opacity=0.6)
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No existe la columna Price; esta sección requiere target.")

with tab_train:
    st.markdown("### Entrenamiento (XGBoost + Pipeline)")
    if "Price" not in df.columns:
        st.error("No puedo entrenar: falta la columna 'Price'.")
        st.stop()
    
    run_train = st.button("🚀 Entrenar modelo", type="primary", use_container_width=True)
    
    if run_train:
        with st.spinner("Entrenando..."):
            pipe, metrics = train_model(df, cfg)

        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{metrics['r2']:.4f}")
        c2.metric("RMSE", f"{metrics['rmse']:.2f}")
        c3.metric("MAE", f"{metrics['mae']:.2f}")

        st.write("")
        left, right = st.columns([1.05, 0.95], gap="large")

        with left:
            st.plotly_chart(plot_pred_vs_real(metrics["y_test"], metrics["preds"]), use_container_width=True)

        with right:
            fig_imp = plot_feature_importance(pipe, metrics["X_train"], top_k=25)
            if fig_imp is not None:
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Este modelo no expone feature_importances_. (Raro en XGBRegressor, pero puede pasar).")

        st.session_state["trained_pipe"] = pipe
        st.session_state["train_cfg"] = cfg
        with st.expander("📌 Parámetros de XGBoost (qué hacen y cómo elegirlos)", expanded=False):
            st.markdown("""
        **Idea general:** XGBoost arma muchos árboles pequeños y los va combinando.  
        La clave es balancear **capacidad** (aprende patrones) vs **regularización** (no se sobreajusta).

        ### Parámetros que estás controlando

        - **n_estimators**: número de árboles.  
          Más árboles = más capacidad, pero también más riesgo de sobreajuste y más tiempo.

        - **max_depth**: profundidad máxima de cada árbol.  
          Más profundo = aprende interacciones complejas, pero sobreajusta más fácil.

        - **learning_rate**: cuánto “aprende” cada árbol nuevo.  
          Valores bajos suelen generalizar mejor, pero requieren más árboles.

        - **subsample**: fracción de filas usada por árbol.  
          < 1.0 mete aleatoriedad y reduce sobreajuste.

        - **colsample_bytree**: fracción de columnas por árbol.  
          Similar a subsample pero con variables.

        - **reg_lambda**: regularización L2.  
          Ayuda a que el modelo no se “pase de listo” con ruido.

        ### Configuración recomendada (muy buena base)
        No existe una “perfecta” universal (depende del dataset), pero como punto de partida sólido:



        - `test_size`: **0.20**
        - `n_estimators`: **200**
        - `max_depth`: **6**
        - `learning_rate`: **0.30**
        - `subsample`: **1.00**
        - `colsample_bytree`: **1.00**
        - `reg_lambda`: **1.00**

        **Regla práctica:**
        - Si ves sobreajuste (R² train alto, test bajo): baja `max_depth`, sube `reg_lambda`, baja `subsample/colsample`.
        - Si el modelo se queda corto (underfit): sube `n_estimators`, sube un poco `max_depth` o baja `reg_lambda`.
        """)

    else:
        st.info("Ajusta parámetros en el sidebar y presiona **Entrenar modelo**.")

with tab_predict:
    st.markdown("### Predicción")
    if "trained_pipe" not in st.session_state:
        st.warning("Primero entrena el modelo en la pestaña **Entrenar**.")
        st.stop()

    pipe = st.session_state["trained_pipe"]

    mode = st.radio("Modo de predicción", ["Individual", "Por lote (CSV)"], horizontal=True)

    feature_df = df.drop(columns=["Price"]) if "Price" in df.columns else df.copy()

    if mode == "Individual":
        st.markdown("#### Configura un auto y predice su precio")

        # UI: inputs dinámicos según tipos
        # UI: inputs dinámicos según tipos + overrides bonitos
        input_data = {}

        # columnas que quieres como ENTERO
        INT_COLS = {"Levy", "Prod. year", "Mileage", "Cylinders", "Airbags"}

        # columnas booleanas
        BOOL_COLS = {"Is_Turbo"}

        cols = st.columns(3)
        all_cols = feature_df.columns.tolist()

        for i, col in enumerate(all_cols):
            with cols[i % 3]:
                
                # 1) Booleanos
                if col in BOOL_COLS:
                    # default basado en la moda, o False si no hay data
                    if col in feature_df.columns and feature_df[col].dropna().shape[0] > 0:
                        default_bool = bool(int(feature_df[col].mode().iloc[0]))
                    else:
                        default_bool = False
                    input_data[col] = st.toggle(col, value=default_bool)

                # 2) Enteros
                elif col in INT_COLS:
                    s = pd.to_numeric(feature_df[col], errors="coerce").dropna()

                    if len(s) > 0:
                        vmin = int(np.floor(s.quantile(0.01)))
                        vmax = int(np.ceil(s.quantile(0.99)))
                        default = int(np.round(s.median()))
                        # evita min>max por si hay poca varianza
                        if vmin > vmax:
                            vmin, vmax = int(s.min()), int(s.max())
                        input_data[col] = st.number_input(
                            col,
                            value=default,
                            min_value=vmin,
                            max_value=vmax,
                            step=1,
                        )
                    else:
                        input_data[col] = st.number_input(col, value=0, step=1)

                # 3) Numéricos (float)
                elif pd.api.types.is_numeric_dtype(feature_df[col]):
                    s = pd.to_numeric(feature_df[col], errors="coerce").dropna()
                    if len(s) > 0:
                        vmin, vmax = float(s.quantile(0.01)), float(s.quantile(0.99))
                        default = float(s.median())
                        input_data[col] = st.number_input(
                            col,
                            value=default,
                            min_value=vmin,
                            max_value=vmax,
                        )
                    else:
                        input_data[col] = st.number_input(col, value=0.0)

                # 4) Categóricas
                else:
                    options = sorted(feature_df[col].dropna().astype(str).unique().tolist())
                    input_data[col] = st.selectbox(col, options=options if options else [""])

        one = pd.DataFrame([input_data])

        # Convertimos bool a 0/1 porque tu pipeline espera numérico
        if "Is_Turbo" in one.columns:
            one["Is_Turbo"] = one["Is_Turbo"].astype(int)

        st.write("")
        do_pred = st.button("🔮 Predecir precio", type="primary", use_container_width=True)

        if do_pred:
            pred = float(pipe.predict(one)[0])

            st.markdown(
                f"""
                <div style='background: rgba(90,120,255,0.14); border: 1px solid rgba(90,120,255,0.35);
                     border-radius: 18px; padding: 16px; font-size: 1.1rem;'>
                    💰 <b>Precio estimado:</b>
                  <span style='font-size: 1.55rem; margin-left: 8px;'>${pred:,.0f}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # opcional: ver la fila que se usó para predecir
            with st.expander("Ver inputs usados para la predicción"):
                st.dataframe(one, use_container_width=True)
        else:
            st.info("Ajusta los campos y presiona **Predecir precio**.")
            

    else:
        st.markdown("#### Predicción por lote")
        st.caption("Sube un CSV con las mismas columnas (sin Price). Te devuelvo otro CSV con predicciones.")
        batch = st.file_uploader("CSV para predecir", type=["csv"], key="batch_uploader")

        if batch is not None:
            batch_raw = pd.read_csv(batch)
            batch_df = clean_dataframe(batch_raw)

            # aplicar el mismo grouping con los mismos thresholds del sidebar
            for col, q in [
                ("Manufacturer", q_man),
                ("Model", q_model),
                ("Category", q_cat),
                ("Fuel type", q_fuel),
                ("Color", q_color),
            ]:
                if col in batch_df.columns:
                    batch_df = group_rare_categories(batch_df, col, q, other_label="Other")

            # quitar Price si viene
            if "Price" in batch_df.columns:
                batch_df = batch_df.drop(columns=["Price"])

            preds = pipe.predict(batch_df)
            out = batch_raw.copy()
            out["Predicted_Price"] = preds

            st.success("Listo ✅")
            st.dataframe(out.head(30), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar CSV con predicciones",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
