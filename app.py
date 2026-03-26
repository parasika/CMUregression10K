import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Myopic Regression Predictor",
    page_icon="👁️",
    layout="wide"
)

MODEL_PATH = "myopia_10param_model.pkl"

@st.cache_resource
def load_bundle():
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        return None
    return joblib.load(model_file)

bundle = load_bundle()

if bundle is None:
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = bundle["model"]
imputer = bundle["imputer"]
features = bundle["features"]

st.title("👁️ Myopic Regression Prediction Model")
st.write("Predict the probability of myopic regression using the 10-parameter model.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Clinical Parameters")
    PRK = st.selectbox("PRK (0 = No, 1 = Yes)", [0, 1])
    Preop_SE_calc = st.number_input("Preop_SE_calc", value=-4.50, format="%.2f")
    Ablation_depth = st.number_input("Ablation_depth", value=80.0, format="%.2f")
    AGE = st.number_input("AGE", value=25.0, format="%.1f")

with col2:
    st.subheader("Corneal Parameters")
    ACD = st.number_input("ACD", value=3.20, format="%.2f")
    K2_B = st.number_input("K2_B", value=6.50, format="%.2f")
    Pachy_Min = st.number_input("Pachy_Min", value=520.0, format="%.1f")
    TBI = st.number_input("TBI", value=0.30, format="%.2f")
    A1_Time_ms = st.number_input("A1_Time_ms", value=7.20, format="%.2f")
    ARTh = st.number_input("ARTh", value=400.0, format="%.1f")

if st.button("🔍 Predict Probability", use_container_width=True):
    input_df = pd.DataFrame([{
        "PRK": PRK,
        "Preop_SE_calc": Preop_SE_calc,
        "Ablation_depth": Ablation_depth,
        "ACD": ACD,
        "K2_B": K2_B,
        "Pachy_Min": Pachy_Min,
        "TBI": TBI,
        "A1_Time_ms": A1_Time_ms,
        "ARTh": ARTh,
        "AGE": AGE
    }])

    try:
        input_df = input_df[features]

        input_imp = pd.DataFrame(
            imputer.transform(input_df),
            columns=features
        )

        prob = float(model.predict_proba(input_imp)[0, 1])

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

        st.write(f"**Exact probability = {prob:.4f}**")

        if prob >= 0.7:
            st.error("🔴 High Risk")
        elif prob >= 0.5:
            st.warning("🟠 Moderate Risk")
        elif prob >= 0.3:
            st.info("🟡 Intermediate Risk")
        else:
            st.success("🟢 Low Risk")

        st.subheader("Input Data")
        st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)