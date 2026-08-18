"""
Spam Mail Classifier — Streamlit App

UI only. Every bit of modeling logic — training, evaluation, and the
predict_and_explain / predict_with_both_models functions themselves —
lives in the notebook (spam_classifier_enhanced.ipynb), which exports
them via cloudpickle to spam_classifier_artifacts.pkl. This file just
loads that bundle and wires it up to Streamlit widgets.
"""

import streamlit as st
import sys

st.write("Python:", sys.version)

try:
    import cloudpickle
    st.write("Cloudpickle:", cloudpickle.__version__)
except Exception as e:
    st.write("Cloudpickle error:", e)
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px



@st.cache_resource
def load_artifacts():
    with open("spam_classifier_artifacts.pkl", "rb") as f:
        return cloudpickle.load(f)


artifacts = load_artifacts()
predict_and_explain = artifacts["predict_and_explain"]
predict_with_both_models = artifacts["predict_with_both_models"]
lr_metrics = artifacts["lr_metrics"]
nb_metrics = artifacts["nb_metrics"]



st.title("Spam Mail Classifier")

message = st.text_area("Enter a message to check:", height=120)

col_a, col_b = st.columns(2)
with col_a:
    analyze_clicked = st.button("Analyze Message")
with col_b:
    compare_clicked = st.button("Compare Models")

if analyze_clicked:
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        label, confidence, probability, contrib_df = predict_and_explain(message)

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
        st.write(f"Spam probability: **{probability[0] * 100:.2f}%**")
        st.write(f"Not Spam probability: **{probability[1] * 100:.2f}%**")

        st.subheader("Why this prediction?")
        if contrib_df is None:
            st.info("No recognizable words to explain in this message.")
        else:
            fig, ax = plt.subplots(figsize=(8, max(3, len(contrib_df) * 0.4)))
            colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in contrib_df['contribution']]
            ax.barh(contrib_df['word'], contrib_df['contribution'], color=colors)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_title("Word Contributions (red = toward spam, green = toward ham)")
            st.pyplot(fig)

if compare_clicked:
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        # -----------------------------------------------------------
        # Side-by-side comparison: what each model says about THIS message
        # -----------------------------------------------------------
        st.subheader("Model Comparison — Same Message, Both Models")

        both_results = predict_with_both_models(message)
        col1, col2 = st.columns(2)

        for col, model_name in zip([col1, col2], both_results.keys()):
            result = both_results[model_name]
            prob = result["probability"]
            with col:
                st.markdown(f"**{model_name}**")
                st.write(f"Prediction: **{result['label']}**")
                st.write(f"Spam probability: {prob[0] * 100:.2f}%")
                st.write(f"Not Spam probability: {prob[1] * 100:.2f}%")

                pie_df = pd.DataFrame({
                    "Class": ["Spam", "Not Spam"],
                    "Probability": [prob[0], prob[1]],
                })
                pie_fig = px.pie(
                    pie_df,
                    names="Class",
                    values="Probability",
                    color="Class",
                    color_discrete_map={"Spam": "#e74c3c", "Not Spam": "#2ecc71"},
                    hole=0.35,
                )
                pie_fig.update_traces(textinfo='label+percent')
                pie_fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=300,
                    showlegend=False,
                )
                st.plotly_chart(pie_fig, use_container_width=True)

        # -----------------------------------------------------------
        # Overall test-set performance comparison (accuracy/precision/recall/F1)
        # -----------------------------------------------------------
        st.markdown("---")
        st.subheader("Model Comparison — Overall Test Performance")

        metric_names = list(lr_metrics.keys())
        metrics_df = pd.DataFrame({
            "Metric": metric_names * 2,
            "Score": [lr_metrics[m] for m in metric_names] + [nb_metrics[m] for m in metric_names],
            "Model": ["Logistic Regression"] * len(metric_names) + ["Naive Bayes"] * len(metric_names),
        })

        bar_fig = px.bar(
            metrics_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            color_discrete_map={"Logistic Regression": "#3498db", "Naive Bayes": "#f39c12"},
            title="Logistic Regression vs Naive Bayes — Test Set Metrics",
        )
        bar_fig.update_layout(
            yaxis=dict(range=[0, 1.05]),
            margin=dict(l=10, r=10, t=40, b=10),
            height=420,
        )
        st.plotly_chart(bar_fig, use_container_width=True)
