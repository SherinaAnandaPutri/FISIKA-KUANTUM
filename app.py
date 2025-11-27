import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="1D Infinite Well ML Predictor",
    page_icon="⭐",
    layout="wide"
)

# -----------------------------
# Load RF Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("rf_infinite_well_model.joblib")

model = load_model()

# -----------------------------
# Analytic energy
# -----------------------------
def analytic_energy(L, n=1):
    hbar = 1.0545718e-34
    m = 9.10938356e-31
    L_m = L * 1e-9
    En = (n**2 * np.pi**2 * hbar**2) / (2 * m * L_m**2)
    return En / 1.60218e-19

# -----------------------------
# Wavefunction
# -----------------------------
def psi_n(x, L, n=1):
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

# =============================
# MAIN UI
# =============================
st.title("🔮 Machine Learning Predictor – 1D Infinite Potential Well")
st.caption("Prediksi Energi Tingkat Dasar dengan Random Forest + Visualisasi Fungsi Gelombang")

st.sidebar.header("⚙️ Parameter Input")
L = st.sidebar.slider("Lebar Sumur (nm)", 0.1, 5.0, 1.0, 0.1)
n = st.sidebar.slider("Tingkat Energi (n)", 1, 5, 1)

st.divider()

# -----------------------------
# Predictions
# -----------------------------
ml_pred = model.predict(np.array([[L]]))[0]
analytic = analytic_energy(L, 1)

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚡ Prediksi Machine Learning (n=1)")
    st.metric("Energi (eV)", f"{ml_pred:.6f}")

with col2:
    st.subheader("📘 Energi Analitik (n=1)")
    st.metric("Energi (eV)", f"{analytic:.6f}")

st.divider()

# -----------------------------
# Wavefunction & Probability
# -----------------------------
st.subheader("🌊 Fungsi Gelombang ψ(x) & Densitas Probabilitas |ψ|²")

x = np.linspace(0, L, 500)
psi = psi_n(x, L, n)
prob = psi**2

col3, col4 = st.columns(2)

with col3:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(x, psi, linewidth=2)
    ax1.set_title(f"Wavefunction ψₙ(x) (n={n})")
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("ψ(x)")
    ax1.grid(True)
    st.pyplot(fig1)

with col4:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(x, prob, linewidth=2)
    ax2.set_title(f"Densitas Probabilitas |ψ|² (n={n})")
    ax2.set_xlabel("x (nm)")
    ax2.set_ylabel("|ψ|²")
    ax2.grid(True)
    st.pyplot(fig2)

st.divider()

# -----------------------------
# Energy vs L
# -----------------------------
st.subheader("📉 Grafik Energi vs Lebar Sumur (Analitik)")

L_vals = np.linspace(0.1, 5, 300)
E_vals = np.array([analytic_energy(Li) for Li in L_vals])

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.plot(L_vals, E_vals, linewidth=2)
ax3.set_title("Energi Tingkat Dasar vs Lebar Sumur")
ax3.set_xlabel("L (nm)")
ax3.set_ylabel("E (eV)")
ax3.grid(True)

st.pyplot(fig3)

st.success("✨ Simulasi selesai! Silakan ubah parameter untuk melihat efeknya.")
