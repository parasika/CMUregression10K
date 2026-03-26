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
    max-width: 1180px;
}
.banner {
    background: linear-gradient(135deg, #e9d5ff 0%, #c084fc 35%, #a855f7 70%, #7e22ce 100%);
    padding: 30px 34px;
    border-radius: 24px;
    color: white;
    box-shadow: 0 14px 32px rgba(126, 34, 206, 0.22);
    margin-bottom: 24px;
}
.banner-title {
    margin: 0;
    font-size: 2.15rem;
    font-weight: 800;
    line-height: 1.2;
}
.banner-subtitle {
    margin-top: 8px;
    margin-bottom: 0;
    font-size: 1.02rem;
    opacity: 0.97;
}
.card {
    background: #ffffff;
    border: 1px solid #eadcff;
    border-radius: 20px;
    padding: 18px 18px 14px 18px;
    margin-bottom: 16px;
    box-shadow: 0 8px 18px rgba(168, 85, 247, 0.08);
}
.soft-card {
    background: #fcfaff;
    border: 1px solid #eadcff;
    border-radius: 20px;
    padding: 18px 18px 14px 18px;
    margin-bottom: 16px;
    box-shadow: 0 8px 18px rgba(168, 85, 247, 0.07);
}
.prob-box {
    background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 45%, #9333ea 100%);
    color: white;
    padding: 22px;
    border-radius: 20px;
    text-align: center;
    font-size: 2.35rem;
    font-weight: 800;
    margin-top: 4px;
    margin-bottom: 14px;
    box-shadow: 0 10px 22px rgba(109, 40, 217, 0.22);
}
.tag-high {
    background: #fee2e2;
    color: #991b1b;
    padding: 10px 14px;
    border-radius: 12px;
    font-weight: 700;
    display: inline-block;
}
.tag-mod {
    background: #ffedd5;
    color: #9a3412;
    padding: 10px 14px;
    border-radius: 12px;
    font-weight: 700;
    display: inline-block;
}
.tag-mid {
    background: #fef9c3;
    color: #854d0e;
    padding: 10px 14px;
    border-radius: 12px;
    font-weight: 700;
    display: inline-block;
}
.tag-low {
    background: #dcfce7;
    color: #166534;
    padding: 10px 14px;
    border-radius: 12px;
    font-weight: 700;
    display: inline-block;
}
.explain-item {
    background: #f8f4ff;
    border-left: 5px solid #9333ea;
    padding: 10px 12px;
    border-radius: 10px;
    margin-bottom: 8px;
    color: #3b0764;
    font-weight: 500;
}
.small-note {
    color: #6b7280;
    font-size: 0.92rem;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #7c3aed 0%, #9333ea 100%);
    color: white;
    font-weight: 700;
    border: none;
    border-radius: 14px;
    padding: 0.75rem 1rem;
    box-shadow: 0 8px 18px rgba(147, 51, 234, 0.22);
}
div[data-testid="stButton"] button:hover {
    filter: brightness(1.03);
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

def unpack_bundle(bundle):
    """
    Supports:
    1) dict: {"model":..., "imputer":..., "features":...}
    2) tuple/list: (model, imputer, features)
    3) plain sklearn model/pipeline
    """
    model = None
    imputer = None
    features = None

    if isinstance(bundle, dict):
        model = bundle.get("model", bundle.get("clf", bundle.get("pipeline", None)))
        imputer = bundle.get("imputer", None)
        features = bundle.get("features", bundle.get("feature_names", None))

    elif isinstance(bundle, (list, tuple)):
        if len(bundle) >= 1:
            model = bundle[0]
        if len(bundle) >= 2:
            imputer = bundle[1]
        if len(bundle) >= 3:
            features = bundle[2]

    else:
        model = bundle

    return model, imputer, features

bundle = load_bundle()

if bundle is None:
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model, imputer, features = unpack_bundle(bundle)

if model is None:
    st.error("Could not read model from the PKL file.")
    st.write("Loaded object type:", type(bundle))
    st.stop()

# fallback feature list if not stored in pkl
DEFAULT_FEATURES = [
    "PRK",
    "Preop_SE_calc",
    "Ablation_depth",
    "ACD",
    "K2_B",
    "Pachy_Min",
    "CBI",
    "A1_Time_ms",
    "ARTh",
    "AGE"
]

if features is None:
    features = DEFAULT_FEATURES

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
            "bar": {"color": "#3b0764", "thickness": 0.34},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#ddd6fe",
            "steps": [
                {"range": [0, 30], "color": "#16a34a"},
                {"range": [30, 50], "color": "#eab308"},
                {"range": [50, 70], "color": "#f97316"},
                {"range": [70, 100], "color": "#dc2626"},
            ],
            "threshold": {
                "line": {"color": "#111827", "width": 5},
                "thickness": 0.8,
                "value": 50
            }
        }
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=18, r=18, t=65, b=18),
        paper_bgcolor="white",
        font={"color": "#111827"}
    )
    return fig

def get_risk_style(prob):
    if prob >= 0.7:
        return "🔴 High Risk", "tag-high"
    elif prob >= 0.5:
        return "🟠 Moderate Risk", "tag-mod"
    elif prob >= 0.3:
        return "🟡 Intermediate Risk", "tag-mid"
    else:
        return "🟢 Low Risk", "tag-low"

def simple_explanation(PRK, Preop_SE_calc, Ablation_depth, AGE, ACD, K2_B, Pachy_Min, CBI, A1_Time_ms, ARTh):
    reasons = []
    if CBI > 0.50:
        reasons.append("High CBI")
    if Ablation_depth > 100:
        reasons.append("Deep ablation depth")
    if Preop_SE_calc < -6.00:
        reasons.append("High pre-operative myopia")
    if ARTh < 300:
        reasons.append("Low ARTh suggesting weaker corneal biomechanical profile")
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

def predict_probability(model, input_df, imputer=None):
    X = input_df.copy()

    if imputer is not None:
        X = pd.DataFrame(imputer.transform(X), columns=X.columns)

    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1])

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        return 1 / (1 + (2.718281828 ** (-score)))

    pred = float(model.predict(X)[0])
    return pred

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="banner">
    <div class="banner-title">🟣👁️ CMU Myopic Regression Prediction Model</div>
    <div class="banner-subtitle">Estimate the probability of myopic regression using clinical, corneal tomography, and corneal biomechanics parameters.</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# MAIN LAYOUT
# =========================================================
left_col, right_col = st.columns([1.05, 1], gap="large")

with left_col:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("🩺 Clinical Parameters")
    PRK_label = st.selectbox("PRK", ["No", "Yes"])
    PRK = 1 if PRK_label == "Yes" else 0
    Preop_SE_calc = st.number_input("Pre-op SE", value=-4.50, format="%.2f")
    Ablation_depth = st.number_input("Ablation depth", value=80.0, format="%.2f")
    AGE = st.number_input("Age", value=25.0, format="%.1f")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("📷 Corneal Tomography")
    ACD = st.number_input("ACD", value=3.20, format="%.2f")
    K2_B = st.number_input("K2 (back)", value=6.50, format="%.2f")
    Pachy_Min = st.number_input("Thinnest Pachy.", value=520.0, format="%.1f")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Corneal Biomechanics")
    CBI = st.number_input("CBI", value=0.30, format="%.2f")
    A1_Time_ms = st.number_input("A1 time", value=7.20, format="%.2f")
    ARTh = st.number_input("ARTh", value=400.0, format="%.1f")
    st.markdown('</div>', unsafe_allow_html=True)

    predict_button = st.button("🔮 Predict Probability", use_container_width=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")
    st.markdown('<p class="small-note">Fill in the parameters, then click predict to estimate the risk of myopic regression.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PREDICTION
# =========================================================
if predict_button:
    raw_input_df = pd.DataFrame([{
        "PRK": PRK,
        "Preop_SE_calc": Preop_SE_calc,
        "Ablation_depth": Ablation_depth,
        "ACD": ACD,
        "K2_B": K2_B,
        "Pachy_Min": Pachy_Min,
        "CBI": CBI,
        "A1_Time_ms": A1_Time_ms,
        "ARTh": ARTh,
        "AGE": AGE
    }])

    try:
        missing_cols = [col for col in features if col not in raw_input_df.columns]
        if missing_cols:
            st.error(f"These model features are missing from the app input: {missing_cols}")
            st.stop()

        input_df = raw_input_df[features]
        prob = predict_probability(model, input_df, imputer)

        risk_text, risk_class = get_risk_style(prob)
        reasons = simple_explanation(
            PRK, Preop_SE_calc, Ablation_depth, AGE,
            ACD, K2_B, Pachy_Min, CBI, A1_Time_ms, ARTh
        )

        with right_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Prediction Result")
            st.plotly_chart(make_gauge(prob), use_container_width=True)

            st.markdown(f'<div class="prob-box">{prob*100:.1f}%</div>', unsafe_allow_html=True)
            st.write(f"**Exact probability = {prob:.4f}**")
            st.markdown(f'<div class="{risk_class}">{risk_text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🧠 Simple Risk Explanation")
            if reasons:
                for reason in reasons:
                    st.markdown(f'<div class="explain-item">• {reason}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="explain-item">• No obvious high-risk pattern detected from the simple rule-based explanation.</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="small-note">This explanation is a simple supportive summary and not a SHAP-based individual feature attribution.</p>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📋 Input Summary")
            display_df = pd.DataFrame([{
                "PRK": PRK_label,
                "Pre-op SE": Preop_SE_calc,
                "Ablation depth": Ablation_depth,
                "Age": AGE,
                "ACD": ACD,
                "K2 (back)": K2_B,
                "Thinnest Pachy.": Pachy_Min,
                "CBI": CBI,
                "A1 time": A1_Time_ms,
                "ARTh": ARTh
            }])
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error("Prediction failed")
        st.write("Loaded bundle type:", type(bundle))
        st.write("Loaded model type:", type(model))
        st.write("Features:", features)
        st.exception(e)