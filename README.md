# 📧 Spam Mail Classifier

A machine learning project that classifies messages as **Spam** or **Not Spam** using Natural Language Processing (NLP).

The project uses **TF-IDF** for text feature extraction and **Logistic Regression** as the primary classification model, with **Multinomial Naive Bayes** used for comparison. The project also includes word-level explainability to show which words contributed to a prediction.

## 🔗 Live Links

* 🚀 **Live Demo:** https://email-spam-detection-enrichment.streamlit.app/
* 💻 **GitHub Repository:** https://github.com/praful011-nep/E-mail-Spam-Detection.git

---

## 📌 Project Overview

The project follows a complete machine learning workflow:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Train / Test Split
   ↓
TF-IDF Feature Extraction
   ↓
Logistic Regression
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Naive Bayes Comparison
   ↓
Prediction & Explainability
   ↓
Streamlit Application
```

The trained model is used by the Streamlit application to classify user-provided messages.

---

## ✨ Features

*  Spam and Not Spam classification
*  TF-IDF text feature extraction
*  Logistic Regression classification
*  Multinomial Naive Bayes comparison
*  Hyperparameter tuning with `GridSearchCV`
*  Model evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix
  * Classification Report
*  Word-level prediction explainability
*  Interactive probability visualization
*  Streamlit web application

---

## 🧠 Machine Learning Approach

### TF-IDF Vectorization

The text messages are converted into numerical features using `TfidfVectorizer`.

The vectorizer uses:

* English stop-word removal
* Lowercase text
* Unigrams and bigrams
* Minimum document frequency of 1

```python
TfidfVectorizer(
    min_df=1,
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2)
)
```

### Logistic Regression

Logistic Regression is used as the primary classification model.

Hyperparameter tuning is performed using `GridSearchCV` with different values of `C`, with the model optimized using **F1-score**.

### Multinomial Naive Bayes

A Multinomial Naive Bayes classifier is also trained using the TF-IDF features and evaluated against the Logistic Regression model.

---

## 💡 Model Explainability

The application provides word-level explanations for predictions.

For Logistic Regression, word contributions are determined using the relationship between the TF-IDF feature values and the model's learned coefficients:

```text
Word Contribution = TF-IDF Value × Model Coefficient
```

This makes it possible to identify which words contributed toward the **Spam** or **Not Spam** prediction.

---

## 📊 Model Evaluation

The models are evaluated using multiple classification metrics:

| Metric           | Description                                 |
| ---------------- | ------------------------------------------- |
| Accuracy         | Overall percentage of correct predictions   |
| Precision        | Correctness of positive predictions         |
| Recall           | Ability to identify positive samples        |
| F1-score         | Balance between precision and recall        |
| Confusion Matrix | Shows correct and incorrect classifications |

The models are compared based on their performance, with particular attention given to the **Spam-class F1-score**.

---

## 📁 Project Structure

```text
Spam-Mail-Classifier/
│
├── Spam-Mail-Classifier.ipynb
│
├── datasets/
│   └── mail_data.csv
│
├── devcontainer/
│
├── spam_classifier_artifacts.pkl
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Requirements

The project dependencies are listed in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Streamlit**
* **Plotly**
* **Matplotlib**
* **Cloudpickle**
* **Jupyter Notebook**

---

## 📄 License

This project is licensed under the **MIT License**.


