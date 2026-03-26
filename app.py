# =========================================================
# PREDICTION
# =========================================================
if predict_button:
    input_dict = {
        "PRK": PRK,
        "Preop_SE_calc": Preop_SE_calc,
        "Ablation_depth": Ablation_depth,
        "AGE": AGE,
        "ACD": ACD,
        "K2_B": K2_B,
        "Pachy_Min": Pachy_Min,
        "TBI": TBI,
        "A1_Time_ms": A1_Time_ms,
        "ARTh": ARTh,
    }

    input_df = build_input_dataframe(input_dict)

    try:
        probability = predict_risk(model, input_df)
        prediction = int(probability >= THRESHOLD)

        label, bg_color, text_color = risk_label(probability, THRESHOLD)

        with right_col:

            # 🎯 MAIN PROBABILITY DISPLAY
            st.markdown("## 🎯 Predicted Probability")

            st.markdown(
                f"""
                <div style="
                    background-color:#111827;
                    color:white;
                    padding:25px;
                    border-radius:20px;
                    text-align:center;
                    font-size:2.2rem;
                    font-weight:700;
                ">
                    {probability*100:.1f}%
                </div>
                """,
                unsafe_allow_html=True
            )

            # 🔎 Interpretation
            st.markdown(
                f"""
                <div style="
                    background-color:{bg_color};
                    color:{text_color};
                    padding:15px;
                    border-radius:15px;
                    text-align:center;
                    font-size:1.2rem;
                    font-weight:600;
                    margin-top:10px;
                ">
                    {label}
                </div>
                """,
                unsafe_allow_html=True
            )

            # 📊 Gauge (optional but nice)
            st.plotly_chart(make_gauge(probability), use_container_width=True)

            # 🧠 Clinical interpretation
            st.markdown("### 🧠 Clinical Interpretation")

            if probability >= 0.7:
                st.error("Very high risk → consider closer follow-up or undercorrection strategy")
            elif probability >= 0.5:
                st.warning("Moderate–high risk → monitor carefully")
            elif probability >= 0.3:
                st.info("Intermediate risk")
            else:
                st.success("Low risk")

            # 📄 Raw probability (important for papers)
            st.markdown("### 📄 Model Output")
            st.write(f"**Predicted probability of myopic regression = {probability:.4f}**")

            # 📋 Input table
            st.markdown("### 📋 Input Summary")
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)