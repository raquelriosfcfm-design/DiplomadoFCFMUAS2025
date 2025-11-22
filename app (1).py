import streamlit as st
import pandas as pd
import plotly.express as px

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.title("🗺️ Dashboard con Mapas")

# Datos
df = px.data.gapminder().query("year == 2007")[["country", "iso_alpha", "continent"]]
df["valor"] = np.random.randint(10, 100, size=len(df))

fig = px.choropleth(
    df,
    locations="iso_alpha",
    color="valor",
    hover_name="country",
    projection="natural earth",
    title="Mapa Mundial con Valores Aleatorios"
)

st.plotly_chart(fig)
