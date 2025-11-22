
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.title("🤖 Dashboard de Machine Learning")

# Datos sintéticos
X = np.arange(1, 21).reshape(-1,1)
y = 3 * X.flatten() + 10

model = LinearRegression()
model.fit(X, y)

# Gráfica
df = pd.DataFrame({"X": X.flatten(), "y": y})
fig = px.scatter(df, x="X", y="y", trendline="ols", title="Datos y modelo")
st.plotly_chart(fig)

# Predicción
valor = st.number_input("Ingresa valor X para predecir:", min_value=0.0)
pred = model.predict([[valor]])[0]

st.success(f"Predicción del modelo: {pred:.2f}")
