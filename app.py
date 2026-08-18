import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Spam Mail Classifier",
    page_icon="📧",
    layout="wide"
)


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_models():

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    df = pd.read_csv("Dataset/mail_data.csv")

    # --------------------------------------------------------
    # Prepare Dataset
    # --------------------------------------------------------

    data = df.where((pd.notnull(df)), '')

    data["Category"] = data["Category"].astype(object)

    # Spam = 0
    # Ham = 1
    data.loc[data["Category"] == "spam", "Category"] = 0
    data.loc[data["Category"] == "ham", "Category"] = 1

    X = data["Message"]
    Y = data["Category"].astype(int)

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    # --------------------------------------------------------
    # TF-IDF Feature Extraction
    # --------------------------------------------------------

    feature_extraction = TfidfVectorizer(
        min_df=1,
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2)
    )

    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    feature_names = feature_extraction.get_feature_names_out()

    # --------------------------------------------------------
    # Logistic Regression + Hyperparameter Tuning
    # --------------------------------------------------------

    param_grid = {
        "C": [0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        scoring="f1"
    )

    grid_search.fit(X_train_features, Y_train)

    model = grid_search.best_estimator_

    coefficients = model.coef_[0]

    # --------------------------------------------------------
    # Logistic Regression Evaluation
    # --------------------------------------------------------

    lr_pred = model.predict(X_test_features)

    lr_accuracy = accuracy_score(Y_test, lr_pred)
    lr_precision = precision_score(
        Y_test,
        lr_pred,
        pos_label=0
    )
    lr_recall = recall_score(
        Y_test,
        lr_pred,
        pos_label=0
    )
    lr_f1 = f1_score(
        Y_test,
        lr_pred,
        pos_label=0
    )

    # --------------------------------------------------------
    # Naive Bayes
    # --------------------------------------------------------

    nb_model = MultinomialNB()

    nb_model.fit(
        X_train_features,
        Y_train
    )

    nb_pred = nb_model.predict(
        X_test_features
    )

    nb_accuracy = accuracy_score(
        Y_test,
        nb_pred
    )

    nb_precision = precision_score(
        Y_test,
        nb_pred,
        pos_label=0
    )

    nb_recall = recall_score(
        Y_test,
        nb_pred,
        pos_label=0
    )

    nb_f1 = f1_score(
        Y_test,
        nb_pred,
        pos_label=0
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    lr_metrics = {
        "Accuracy": lr_accuracy,
        "Precision (spam)": lr_precision,
        "Recall (spam)": lr_recall,
        "F1-score (spam)": lr_f1
    }

    nb_metrics = {
        "Accuracy": nb_accuracy,
        "Precision (spam)": nb_precision,
        "Recall (spam)": nb_recall,
        "F1-score (spam)": nb_f1
    }

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Multinomial Naive Bayes"
        ],
        "Accuracy": [
            lr_accuracy,
            nb_accuracy
        ],
        "Precision (spam)": [
            lr_precision,
            nb_precision
        ],
        "Recall (spam)": [
            lr_recall,
            nb_recall
        ],
        "F1-score (spam)": [
            lr_f1,
            nb_f1
        ]
    })

    return (
        feature_extraction,
        model,
        nb_model,
        feature_names,
        coefficients,
        lr_metrics,
        nb_metrics,
        comparison_df
    )


# ============================================================
# LOAD / TRAIN MODELS
# ============================================================

(
    feature_extraction,
    model,
    nb_model,
    feature_names,
    coefficients,
    lr_metrics,
    nb_metrics,
    comparison_df
) = train_models()


# ============================================================
# PREDICTION + EXPLANATION
# ============================================================

def predict_and_explain(message, top_n=8):

    features = feature_extraction.transform([message])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0]

    label = "Spam" if prediction == 0 else "Not Spam"

    confidence = (
        probability[0]
        if prediction == 0
        else probability[1]
    )

    feature_array = features.toarray()[0]

    present_indices = feature_array.nonzero()[0]

    contrib_df = None

    if len(present_indices) > 0:

        contributions = []

        for idx in present_indices:

            word = feature_names[idx]

            tfidf_val = feature_array[idx]

            coef = coefficients[idx]

            contributions.append({
                "word": word,
                "contribution": tfidf_val * coef
            })

        contrib_df = pd.DataFrame(
            contributions
        ).sort_values(
            "contribution"
        )

        contrib_df = pd.concat([
            contrib_df.head(top_n),
            contrib_df.tail(top_n)
        ]).drop_duplicates()

    return (
        label,
        confidence,
        probability,
        contrib_df
    )


# ============================================================
# BOTH MODEL PREDICTION
# ============================================================

def predict_with_both_models(message):

    features = feature_extraction.transform([message])

    # Logistic Regression

    lr_probability = model.predict_proba(
        features
    )[0]

    lr_pred = model.predict(
        features
    )[0]

    # Naive Bayes

    nb_probability = nb_model.predict_proba(
        features
    )[0]

    nb_pred = nb_model.predict(
        features
    )[0]

    return {

        "Logistic Regression": {
            "label": (
                "Spam"
                if lr_pred == 0
                else "Not Spam"
            ),
            "probability": lr_probability
        },

        "Multinomial Naive Bayes": {
            "label": (
                "Spam"
                if nb_pred == 0
                else "Not Spam"
            ),
            "probability": nb_probability
        }
    }


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("📧 Spam Mail Classifier")

st.write(
    "Enter a message below to determine whether it is "
    "Spam or Not Spam."
)

message = st.text_area(
    "Enter a message to check:",
    height=120
)

col_a, col_b = st.columns(2)

with col_a:

    analyze_clicked = st.button(
        "Analyze Message",
        use_container_width=True
    )

with col_b:

    compare_clicked = st.button(
        "Compare Models",
        use_container_width=True
    )


# ============================================================
# ANALYZE MESSAGE
# ============================================================

if analyze_clicked:

    if not message.strip():

        st.warning(
            "Please enter a message first."
        )

    else:

        (
            label,
            confidence,
            probability,
            contrib_df
        ) = predict_and_explain(message)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        st.subheader(
            f"Prediction: {label}"
        )

        st.write(
            f"Confidence: "
            f"**{confidence * 100:.2f}%**"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Spam Probability",
                f"{probability[0] * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Not Spam Probability",
                f"{probability[1] * 100:.2f}%"
            )

        # ----------------------------------------------------
        # Why this prediction?
        # ----------------------------------------------------

        st.subheader(
            "Why this prediction?"
        )

        if contrib_df is None:

            st.info(
                "No recognizable words to explain "
                "in this message."
            )

        else:

            fig, ax = plt.subplots(
                figsize=(
                    8,
                    max(3, len(contrib_df) * 0.4)
                )
            )

            colors = [
                "#e74c3c"
                if value < 0
                else "#2ecc71"
                for value in contrib_df[
                    "contribution"
                ]
            ]

            ax.barh(
                contrib_df["word"],
                contrib_df["contribution"],
                color=colors
            )

            ax.axvline(
                0,
                color="black",
                linewidth=0.8
            )

            ax.set_title(
                "Word Contributions "
                "(red = toward spam, "
                "green = toward ham)"
            )

            st.pyplot(fig)


# ============================================================
# MODEL COMPARISON
# ============================================================

if compare_clicked:

    if not message.strip():

        st.warning(
            "Please enter a message first."
        )

    else:

        st.subheader(
            "Model Comparison — "
            "Same Message, Both Models"
        )

        both_results = predict_with_both_models(
            message
        )

        col1, col2 = st.columns(2)

        for col, model_name in zip(
            [col1, col2],
            both_results.keys()
        ):

            result = both_results[
                model_name
            ]

            prob = result[
                "probability"
            ]

            with col:

                st.markdown(
                    f"### {model_name}"
                )

                st.write(
                    f"Prediction: "
                    f"**{result['label']}**"
                )

                st.write(
                    f"Spam probability: "
                    f"{prob[0] * 100:.2f}%"
                )

                st.write(
                    f"Not Spam probability: "
                    f"{prob[1] * 100:.2f}%"
                )

                pie_df = pd.DataFrame({
                    "Class": [
                        "Spam",
                        "Not Spam"
                    ],
                    "Probability": [
                        prob[0],
                        prob[1]
                    ]
                })

                pie_fig = px.pie(
                    pie_df,
                    names="Class",
                    values="Probability",
                    color="Class",
                    color_discrete_map={
                        "Spam": "#e74c3c",
                        "Not Spam": "#2ecc71"
                    },
                    hole=0.35
                )

                pie_fig.update_traces(
                    textinfo="label+percent"
                )

                pie_fig.update_layout(
                    margin=dict(
                        l=10,
                        r=10,
                        t=10,
                        b=10
                    ),
                    height=300,
                    showlegend=False
                )

                st.plotly_chart(
                    pie_fig,
                    use_container_width=True
                )

        # ----------------------------------------------------
        # Overall Model Performance
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "Model Comparison — "
            "Overall Test Performance"
        )

        metric_names = list(
            lr_metrics.keys()
        )

        metrics_df = pd.DataFrame({

            "Metric": (
                metric_names * 2
            ),

            "Score": (
                [
                    lr_metrics[m]
                    for m in metric_names
                ]
                +
                [
                    nb_metrics[m]
                    for m in metric_names
                ]
            ),

            "Model": (
                [
                    "Logistic Regression"
                ] * len(metric_names)
                +
                [
                    "Naive Bayes"
                ] * len(metric_names)
            )
        })

        bar_fig = px.bar(
            metrics_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            color_discrete_map={
                "Logistic Regression": "#3498db",
                "Naive Bayes": "#f39c12"
            },
            title=(
                "Logistic Regression vs "
                "Naive Bayes — "
                "Test Set Metrics"
            )
        )

        bar_fig.update_layout(
            yaxis=dict(
                range=[0, 1.05]
            ),
            margin=dict(
                l=10,
                r=10,
                t=40,
                b=10
            ),
            height=420
        )

        st.plotly_chart(
            bar_fig,
            use_container_width=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.subheader("Model Performance")

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)

best_model_name = comparison_df.loc[
    comparison_df["F1-score (spam)"].idxmax(),
    "Model"
]

st.success(
    f"Model with the higher F1-score: "
    f"**{best_model_name}**"
)