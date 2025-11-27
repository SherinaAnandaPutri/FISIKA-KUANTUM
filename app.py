import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Simulasi Fisika Kuantum", layout="wide")

st.title("🔷 Simulasi Fisika Kuantum: 1D Infinite Potential Well")

st.write("""
Simulasi ini memodelkan fungsi gelombang partikel dalam sumur potensial tak-hingga 1 dimensi.
Persamaan yang digunakan:

- Fungsi gelombang:  
  \\( \psi_n(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi x}{L}\right) \\)

- Energi:  
  \\( E_n = \frac{n^2 \pi^2 \hbar^2}{2mL^2} \\)
""")

# Sidebar
st.sidebar.header("⚙️ Pengaturan Simulasi")
L = st.sidebar.slider("Panjang Sumur (L)", 1.0, 10.0, 5.0)
n = st.sidebar.slider("Level Energi (n)", 1, 10, 1)
num_points = st.sidebar.slider("Resolusi Grafik", 200, 2000, 800)

# Generate x values
x = np.linspace(0, L, num_points)

# Wave function
psi = np.sqrt(2 / L) * np.sin(n * np.pi * x / L)
prob = psi**2

# Energi
hbar = 1.0545718e-34
m = 9.11e-31
E = (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)

# Plot 1 – Wave Function
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=x, y=psi, mode='lines', name='ψ(x)'))
fig1.update_layout(
    title=f"Fungsi Gelombang ψ(x) untuk n = {n}",
    xaxis_title="x (meter)",
    yaxis_title="ψ(x)",
    template="plotly_white"
)

# Plot 2 – Probability Density
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x, y=prob, mode='lines', name='|ψ(x)|²'))
fig2.update_layout(
    title=f"Densitas Probabilitas |ψ(x)|² untuk n = {n}",
    xaxis_title="x (meter)",
    yaxis_title="|ψ(x)|²",
    template="plotly_white"
)

st.subheader("📈 Fungsi Gelombang")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📊 Densitas Probabilitas")
st.plotly_chart(fig2, use_container_width=True)

st.info(f"🔹 Energi Level ke-{n}: **{E:.3e} Joule**")
