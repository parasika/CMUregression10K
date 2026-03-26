import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CMU Myopic Regression Prediction Model",
    page_icon="🟣",
    layout="wide"
)

MODEL_PATH = "myopia_10param_model.pkl"

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.banner {
    background: linear-gradient(135deg, #c084fc 0%, #a855f7 40%, #7c3aed 100%);
    padding: 28px 32px;
    border-radius: 22px;
    color: white;
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.25);
    margin-bottom: 22px;
}

.banner h1 {
    margin: 0;
    font-size: 2.1rem;
    font-weight: 800;
}

.banner p {
    margin-top: 8px;
    margin-bottom: 0;
    font-size: 1.0rem;
    opacity: 0.96;
}

.section-card {
    background: #faf7ff;
    border: 1px solid #eadcff;
    border-radius: 18px;
    padding: 18px 18px 10px 18px;
    margin-bottom: 16px;
    box-shadow: 0 4px 14px rgba(168, 85, 247, 0.08);
}

.result-card {
    background: #ffffff;
    border: 1px solid #eadcff;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(168, 85, 247, 0.08);
    margin-bottom: 14px;
}

.prob-box {
    background: linear-gradient(135deg, #581c87 0%, #7c3aed 50%, #a855f7 100%);
    color: white;
    padding: 24px;
    border-radius: 18px;
    text-align: center;
    font-size: 2.3rem;
    font-weight: 800;
    margin-bottom: 14px;
    box-shadow: 0 8px 20px rgba(124, 58, 237, 0.22);
}

.small-note {
    color: #6b7280;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL BUNDLE
# =========================================================
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

# =========================================================
# HELPERS
# =========================================================
def make_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 34}},
        title={"text": "Predicted Probability", "font": {"size": 22}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#4c1d95", "thickness": 0.34},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#ddd6fe",
            "steps": [
                {"range": [0, 30], "color": "#22c55e"},   # strong green
                {"range": [30, 50], "color": "#eab308"},  # strong yellow
                {"range": [50, 70], "color": "#f97316"},  # strong orange
                {"range": [70, 100], "color": "#ef4444"}, # strong red
            ],
            "threshold": {
                "line": {"color": "#111827", "width": 5},
                "thickness": 0.8,
                "value": 50
            }
        }
    ))
    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=70, b=20),
        paper_bgcolor="white",
        font={"color": "#111827"}
    )
    return fig


def get_risk_label(prob):
    if prob >= 0.7:
        return "🔴 High Risk"
    elif prob >= 0.5:
        return "🟠 Moderate Risk"
    elif prob >= 0.3:
        return "🟡 Intermediate Risk"
    else:
        return "🟢 Low Risk"


def simple_explanation(PRK, Preop_SE_calc, Ablation_depth, AGE, ACD, K2_B, Pachy_Min, TBI, A1_Time_ms, ARTh):
    reasons = []

    if TBI > 0.50:
        reasons.append("High TBI")
    if Ablation_depth > 100:
        reasons.append("Deep ablation depth")
    if Preop_SE_calc < -6.00:
        reasons.append("High pre-operative myopia")
    if ARTh < 300:
        reasons.append("Low ARTh suggesting weaker corneal structural profile")
    if Pachy_Min < 500:
        reasons.append("Thin thinnest pachymetry")
    if A1_Time_ms < 7.00:
        reasons.append("Short A1 time")
    if PRK == 1:
        reasons.append("PRK treatment profile")
    if AGE < 25:
        reasons.append("Younger age")
    if ACD < 3.0:
        reasons.append("Lower ACD")
    if K2_B > 6.8:
        reasons.append("Steeper posterior corneal curvature")

    return reasons[:5]


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="banner">
    <h1>🟣👁️ CMU Myopic Regression Prediction Model</h1>
    <p>Estimate the probability of myopic regression using clinical, tomographic, and biomechanical parameters.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# LAYOUT
# =========================================================
left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🩺 Clinical Parameters")
    PRK = st.selectbox("PRK", [0, 1], help="0 = No, 1 = Yes")
    Preop_SE_calc = st.number_input("Pre-op SE", value=-4.50, format="%.2f")
    Ablation_depth = st.number_input("Ablation depth", value=80.0, format="%.2f")
    AGE = st.number_input("Age", value=25.0, format="%.1f")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📷 Corneal Tomography")
    ACD = st.number_input("ACD", value=3.20, format="%.2f")
    K2_B = st.number_input("K2 (back)", value=6.50, format="%.2f")
    Pachy_Min = st.number_input("Thinnest Pachy.", value=520.0, format="%.1f")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Corneal Biomechanics")
    TBI = st.number_input("TBI", value=0.30, format="%.2f")
    A1_Time_ms = st.number_input("A1 time", value=7.20, format="%.2f")
    ARTh = st.number_input("ARTh", value=400.0, format="%.1f")
    st.markdown('</div>', unsafe_allow_html=True)

    predict_button = st.button("🔮 Predict Probability", use_container_width=True)

with right_col:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")
    st.markdown('<p class="small-note">Enter the values on the left, then click predict.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PREDICTION
# =========================================================
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
        risk_label = get_risk_label(prob)
        reasons = simple_explanation(
            PRK, Preop_SE_calc, Ablation_depth, AGE,
            ACD, K2_B, Pachy_Min, TBI, A1_Time_ms, ARTh
        )

        with right_col:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("📊 Prediction Result")
            st.plotly_chart(make_gauge(prob), use_container_width=True)

            st.markdown(
                f'<div class="prob-box">{prob*100:.1f}%</div>',
                unsafe_allow_html=True
            )

            st.write(f"**Exact probability = {prob:.4f}**")
            st.write(f"**Risk category: {risk_label}**")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("🧠 Simple Risk Explanation")
            if reasons:
                for r in reasons:
                    st.write(f"- {r}")
            else:
                st.write("- No obvious high-risk pattern detected from the simple rule-based explanation.")
            st.markdown(
                '<p class="small-note">This explanation is a simple supportive summary, not a SHAP-based causal interpretation.</p>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("📋 Input Summary")
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
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)