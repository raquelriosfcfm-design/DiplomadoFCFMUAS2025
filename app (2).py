import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.title("📈 Dashboard de BI Estilo Ejecutivo")

# Datos ficticios
df = pd.DataFrame({
    "mes": ["Ene","Feb","Mar","Abr","May","Jun"],
    "ventas": [12000,15000,17000,16000,18000,20000],
    "clientes": [120,135,140,130,150,160],
})

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Ventas Totales", f"${df['ventas'].sum():,.0f}")
col2.metric("Promedio por Mes", f"${df['ventas'].mean():,.0f}")
col3.metric("Clientes Totales", f"{df['clientes'].sum()}")

# Gráfica de ventas
fig = px.bar(df, x="mes", y="ventas", title="Ventas por Mes")
st.plotly_chart(fig)

# Gráfica de clientes
fig2 = px.line(df, x="mes", y="clientes", markers=True, title="Clientes por Mes")
st.plotly_chart(fig2)
