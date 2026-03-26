import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Myopic Regression Predictor",
    page_icon="👁️",
    layout="wide"
)

MODEL_PATH = "myopia_10param_model.pkl"
THRESHOLD = 0.50

FEATURE_NAMES = [
    "PRK",
    "Preop_SE_calc",
    "Ablation_depth",
    "AGE",
    "ACD",
    "K2_B",
    "Pachy_Min",
    "TBI",
    "A1_Time_ms",
    "ARTh",
]

# =========================================================
# LOAD MODEL (SAFE)
# =========================================================
@st.cache_resource
def load_model():
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        return None
    return joblib.load(model_file)

model = load_model()

# =========================================================
# HEADER
# =========================================================
st.title("👁️ Myopic Regression Prediction Model")

st.markdown("""
Predict the probability of **myopic regression** using a 10-parameter model.
""")

# =========================================================
# CHECK MODEL
# =========================================================
if model is None:
    st.error("❌ Model file not found. Please upload 'myopia_10param_model.pkl'")
    st.stop()

# =========================================================
# INPUT SECTION
# =========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Clinical Parameters")

    PRK = st.selectbox("PRK (0=No, 1=Yes)", [0, 1])
    Preop_SE_calc = st.number_input("Preop SE (calc)", value=-4.50)
    Ablation_depth = st.number_input("Ablation depth", value=80.0)
    AGE = st.number_input("Age", value=25.0)

with col2:
    st.subheader("Corneal Parameters")

    ACD = st.number_input("ACD", value=3.2)
    K2_B = st.number_input("K2 Back", value=6.5)
    Pachy_Min = st.number_input("Pachy Min", value=520.0)
    TBI = st.number_input("TBI", value=0.3)
    A1_Time_ms = st.number_input("A1 Time (ms)", value=7.2)
    ARTh = st.number_input("ARTh", value=400.0)

# =========================================================
# PREDICTION BUTTON
# =========================================================
if st.button("🔍 Predict"):

    # Create dataframe
    input_df = pd.DataFrame([[
        PRK,
        Preop_SE_calc,
        Ablation_depth,
        AGE,
        ACD,
        K2_B,
        Pachy_Min,
        TBI,
        A1_Time_ms,
        ARTh
    ]], columns=FEATURE_NAMES)

    try:
        # Predict probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][1]
        else:
            prob = float(model.predict(input_df)[0])

        # =================================================
        # OUTPUT
        # =================================================
        st.markdown("## 🎯 Predicted Probability")

        st.markdown(
            f"""
            <div style="
                background-color:#111827;
                color:white;
                padding:30px;
                border-radius:20px;
                text-align:center;
                font-size:2.5rem;
                font-weight:700;
            ">
                {prob*100:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        # Risk label
        if prob >= 0.7:
            st.error("🔴 High Risk")
        elif prob >= 0.5:
            st.warning("🟠 Moderate Risk")
        elif prob >= 0.3:
            st.info("🟡 Intermediate Risk")
        else:
            st.success("🟢 Low Risk")

        # Exact probability (important)
        st.write(f"**Exact probability = {prob:.4f}**")

        # Show inputs
        st.subheader("Input Data")
        st.dataframe(input_df)

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)