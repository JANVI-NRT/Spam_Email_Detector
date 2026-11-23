import streamlit as st
import joblib
import numpy as np
import re

# Load trained models
bundle = joblib.load("model_bundle.pkl")

vectorizer = bundle["vectorizer"]
nb_prior = bundle["nb_prior"]
nb_likelihood = bundle["nb_likelihood"]
lr_weights = bundle["lr_weights"]
lr_bias = bundle["lr_bias"]
svm = bundle["svm"]
remove_words = bundle["remove_words"]


def clean(text):
    text = text.lower()
    text = re.sub(r'\d+', ' num ', text)
    text = text.replace('$', ' dollar ')
    text = text.replace('!', ' exclaim ')
    words = re.findall(r'[a-z]+', text)
    words = [w for w in words if w not in remove_words]
    return ' '.join(words)


def predict_email(email):
    cleaned = clean(email)
    X = vectorizer.transform([cleaned]).toarray()

    # Naive Bayes: convert log scores into probabilities
    nb_scores = np.array(nb_prior) + X @ np.array(nb_likelihood).T
    nb_scores = nb_scores.flatten()
    nb_probs = np.exp(nb_scores - np.max(nb_scores))
    nb_probs = nb_probs / np.sum(nb_probs)

    # Logistic Regression: sigmoid probability
    lr_score = float(X @ lr_weights + lr_bias)
    lr_prob = 1 / (1 + np.exp(-np.clip(lr_score, -500, 500)))

    # SVM: probability estimate
    svm_prob = svm.predict_proba(X)[0][1]

    # Average the three spam probability estimates
    spam_probability = np.mean([
        nb_probs[1],
        lr_prob,
        svm_prob
    ])

    return spam_probability, [
        nb_probs[1],
        lr_prob,
        svm_prob
    ]


# Page configuration
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Spam Email Detector")
st.write("Check whether an email is spam or legitimate using machine learning.")

email_text = st.text_area(
    "Paste your email content below:",
    height=250,
    placeholder="Enter the email subject and body here..."
)

if st.button("Classify Email"):
    if email_text.strip():

        spam_probability, model_probs = predict_email(email_text)

        st.subheader("Spam Probability")

        st.metric(
            label="Estimated likelihood of Spam",
            value=f"{spam_probability * 100:.2f}%"
        )

        st.progress(float(spam_probability))

        if spam_probability >= 0.5:
            st.error("🚨 This email is likely SPAM.")
        else:
            st.success("✅ This email is likely NOT SPAM.")

        st.subheader("Individual Model Estimates")

        model_names = [
            "Naive Bayes",
            "Logistic Regression",
            "SVM"
        ]

        for name, prob in zip(model_names, model_probs):
            st.write(f"**{name}:** {prob * 100:.2f}% spam likelihood")

    else:
        st.warning("Please enter an email before classifying.")