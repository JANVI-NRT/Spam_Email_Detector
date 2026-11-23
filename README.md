# Spam Email Detector

An interactive machine learning application that classifies email messages as **Spam** or **Not Spam** using natural language processing and an ensemble of machine learning models.

Users can paste an email into the web application and receive a prediction with estimated spam likelihood.

## Live Demo

**[Try the Spam Email Detector](YOUR_STREAMLIT_APP_URL)**

*Replace the placeholder with your deployed Streamlit URL.*

## Project Overview

Spam emails are commonly used for phishing, fraudulent promotions and unwanted communication. This project explores how machine learning can identify spam by learning patterns in email text.

The project includes:
- Exploratory data analysis of email messages.
- Text preprocessing and feature extraction using TF-IDF.
- Implementation and comparison of multiple classification algorithms.
- A majority-voting ensemble classifier.
- An interactive Streamlit application for testing new emails.

## Dataset and Preprocessing

The project uses a publicly available spam email dataset from Kaggle.

The preprocessing pipeline includes:
- Converting text to lowercase.
- Replacing numerical values and selected symbols.
- Removing dataset-specific words.
- Removing common English stop words through TF-IDF.
- Converting email text into numerical TF-IDF feature vectors.

The TF-IDF vectorizer uses:
- `max_features = 3000`
- `min_df = 5`
- `stop_words = 'english'`

## Machine Learning Models

Four classification algorithms were evaluated:

| Model | Implementation |
|---|---|
| Multinomial Naive Bayes | From scratch |
| Logistic Regression | From scratch |
| Decision Tree | From scratch |
| Support Vector Machine | Scikit-learn |

The final classifier combines Naive Bayes, Logistic Regression and SVM through majority voting. The Decision Tree was excluded from the final ensemble because of its comparatively weaker performance.

## Model Performance

Test-set results from the assignment evaluation:

| Model | Test Accuracy |
|---|---:|
| Naive Bayes | 97.98% |
| Logistic Regression | 98.07% |
| Decision Tree | 94.38% |
| SVM | 98.16% |
| **Ensemble** | **98.16%** |

The ensemble achieved 98.16% test accuracy, with a precision of 98.34% for the spam class.

## Interactive Application

The Streamlit application allows users to:
1. Paste an email message.
2. Submit it for classification.
3. View the predicted category.
4. Inspect individual model estimates and the estimated spam likelihood.

The application loads a previously trained model bundle. Users do not need to upload the original training dataset to classify emails.

## Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Spam-Email-Detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

## Project Structure

```text
Spam-Email-Detector/
│
├── app.py
├── train_model.py
├── model_bundle.pkl
├── requirements.txt
├── Spam_Classifier_FullCodeandResults.ipynb
├── emails.csv
└── test/
```

## Limitations

- The non-spam emails in the training dataset largely represent one company's mailbox, which may affect generalisation to other email styles.
- TF-IDF represents text using word-level features and does not capture word order or context.
- Misspelled or previously unseen words may be difficult to identify.
- Predicted spam likelihoods are model estimates and should not be interpreted as calibrated probabilities.

## Future Improvements

- Evaluate probability calibration.
- Improve robustness to punctuation and formatting changes.
- Test on more diverse email datasets.
- Add model explainability to identify influential words.
- Improve the user interface and deployment experience.

## Technologies Used

Python · NumPy · Pandas · Scikit-learn · TF-IDF · Streamlit · Matplotlib · Seaborn · Git · GitHub

---

**Project developed as part of the Foundations of Machine Learning coursework.**
