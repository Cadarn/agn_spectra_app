"""
Streamlit app to host AGN spectral model generation
Copyright: Adam Hill (2020)
"""

import plotly.express as px
import streamlit as st

from data_functions import generate_spectra

DEGREE_SYMBOL = "\N{DEGREE SIGN}"

st.sidebar.title("AGN Spectral Model Simulator")

# Schematic info
st.sidebar.subheader("AGN Geometry")
st.sidebar.image("assets/schematic.png", use_column_width=True)

# Controllers
inclination = st.sidebar.slider(
    "Inclination angle",
    min_value=0.0,
    max_value=90.0,
    step=0.1,
    format=f"%.1f{DEGREE_SYMBOL}",
    key="incAngle",
)

opening_angle = st.sidebar.slider(
    "Torus opening angle",
    min_value=0.0,
    max_value=90.0,
    step=0.1,
    format=f"%.1f{DEGREE_SYMBOL}",
    key="openingAngle",
)

column_density = st.sidebar.slider(
    "Torus column density, logNH",
    min_value=22.0,
    max_value=25.5,
    step=0.1,
    format="%.1f",
    key="columnDensity",
)

# Add some context in the main window
st.title("AGN Spectral Model Results")
st.subheader("X-ray spectrum")

# generate our data
df = generate_spectra(inclination, opening_angle, column_density)
mod_df = df.melt(
    id_vars=["Energy"], value_vars=["Transmitted", "Reprocessed", "Summed"]
)
mod_df = mod_df.rename(columns={"variable": "Model Component", "value": "Flux"})

# Construct our plot
fig = px.line(
    mod_df,
    x="Energy",
    y="Flux",
    color="Model Component",
    log_x=True,
    width=900,
    height=500,
    labels=dict(Flux="EF_E / keV cm^{-2}s^{-1}", Energy="Energy / keV"),
)
fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

st.plotly_chart(fig)

st.sidebar.markdown("### Model outputs")
if st.sidebar.checkbox("Show Table", False):
    st.subheader("Raw Data Table")
    st.write(df, index=False)

# Some advertising
st.sidebar.markdown("Designed by: Dr Peter Boorman & Dr Adam Hill")
