import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(
    page_title="CMU Myopic Regression Prediction Model",
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

def make_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": "Predicted Probability", "font": {"size": 22}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.3},
            "steps": [
                {"range": [0, 30], "color": "#d1fae5"},
                {"range": [30, 50], "color": "#fde68a"},
                {"range": [50, 70], "color": "#fdba74"},
                {"range": [70, 100], "color": "#fecaca"},
            ],
        }
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    return fig

bundle = load_bundle()

if bundle is None:
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = bundle["model"]
imputer = bundle["imputer"]
features = bundle["features"]

st.title("CMU Myopic Regression Prediction Model")
st.write("Predict the probability of myopic regression using the 10-parameter model.")

left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.subheader("🩺 Clinical Parameters")
    PRK = st.selectbox("PRK", [0, 1], help="0 = No, 1 = Yes")
    Preop_SE_calc = st.number_input("Pre-op SE", value=-4.50, format="%.2f")
    Ablation_depth = st.number_input("Ablation depth", value=80.0, format="%.2f")
    AGE = st.number_input("Age", value=25.0, format="%.1f")

    st.subheader("📷 Corneal Tomography")
    ACD = st.number_input("ACD", value=3.20, format="%.2f")
    K2_B = st.number_input("K2 (back)", value=6.50, format="%.2f")
    Pachy_Min = st.number_input("Thinnest Pachy.", value=520.0, format="%.1f")

    st.subheader("⚙️ Corneal Biomechanics")
    TBI = st.number_input("TBI", value=0.30, format="%.2f")
    A1_Time_ms = st.number_input("A1 time", value=7.20, format="%.2f")
    ARTh = st.number_input("ARTh", value=400.0, format="%.1f")

predict_button = st.button("🔍 Predict Probability", use_container_width=True)

if predict_button:
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

        with right_col:
            st.subheader("📊 Prediction Result")
            st.plotly_chart(make_gauge(prob), use_container_width=True)

            st.markdown(
                f"""
                <div style="
                    background-color:#111827;
                    color:white;
                    padding:22px;
                    border-radius:18px;
                    text-align:center;
                    font-size:2.2rem;
                    font-weight:700;
                    margin-bottom:12px;
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

            st.subheader("📋 Input Data")
            display_df = pd.DataFrame([{
                "PRK": PRK,
                "Pre-op SE": Preop_SE_calc,
                "Ablation depth": Ablation_depth,
                "Age": AGE,
                "ACD": ACD,
                "K2 (back)": K2_B,
                "Thinnest Pachy.": Pachy_Min,
                "TBI": TBI,
                "A1 time": A1_Time_ms,
                "ARTh": ARTh
            }])
            st.dataframe(display_df, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)