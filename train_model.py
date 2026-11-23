
import numpy as np
import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# -----------------------------
# 1. Load and clean dataset
# -----------------------------

df = pd.read_csv("emails.csv")
df = df.drop_duplicates().reset_index(drop=True)

remove_words = [
    'subject', 'enron', 'vince', 'kaminski', 'vkaminski',
    'ect', 'hou', 'houston', 'shirley', 'crenshaw',
    'stinson', 'gibner', 'corp'
]

def clean(text):
    text = text.lower()
    text = re.sub(r'\d+', ' num ', text)
    text = text.replace('$', ' dollar ')
    text = text.replace('!', ' exclaim ')
    words = re.findall(r'[a-z]+', text)
    words = [w for w in words if w not in remove_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean)

X = df['clean_text'].values
y = df['spam'].values

# -----------------------------
# 2. Train-test split
# -----------------------------

np.random.seed(42)

idx = np.arange(len(df))
np.random.shuffle(idx)

split = int(0.8 * len(df))

train = idx[:split]

X_train = X[train]
y_train = y[train]

# -----------------------------
# 3. TF-IDF
# -----------------------------

vec = TfidfVectorizer(
    stop_words='english',
    min_df=5,
    max_features=3000
)

X_train = vec.fit_transform(X_train).toarray()

# Validation split
v = int(0.8 * len(X_train))

X_tr, X_val = X_train[:v], X_train[v:]
y_tr, y_val = y_train[:v], y_train[v:]

# -----------------------------
# 4. Naive Bayes
# -----------------------------

class NaiveBayes:

    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        self.prior = []
        self.likelihood = []

        for c in [0, 1]:
            Xc = X[y == c]
            self.prior.append(np.log(len(Xc) / len(X)))

            word_count = Xc.sum(axis=0) + self.alpha
            self.likelihood.append(
                np.log(word_count / word_count.sum())
            )

    def predict(self, X):
        preds = []

        for x in X:
            score0 = self.prior[0] + np.sum(x * self.likelihood[0])
            score1 = self.prior[1] + np.sum(x * self.likelihood[1])
            preds.append(1 if score1 > score0 else 0)

        return np.array(preds)


nb_scores = {}

for alpha in [0.01, 0.1, 0.5, 1]:
    model = NaiveBayes(alpha)
    model.fit(X_tr, y_tr)
    nb_scores[alpha] = np.mean(model.predict(X_val) == y_val)

best_alpha = max(nb_scores, key=nb_scores.get)

nb = NaiveBayes(best_alpha)
nb.fit(X_train, y_train)

# -----------------------------
# 5. Logistic Regression
# -----------------------------

class LogisticRegression:

    def __init__(self, learning_rate=0.1, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape

        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(self.iterations):
            predictions = self.sigmoid(
                np.dot(X, self.weights) + self.bias
            )

            dw = np.dot(X.T, (predictions - y)) / m
            db = np.sum(predictions - y) / m

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        probabilities = self.sigmoid(
            np.dot(X, self.weights) + self.bias
        )
        return (probabilities >= 0.5).astype(int)


lr_scores = {}

for rate in [1, 5, 10, 20]:
    model = LogisticRegression(
        learning_rate=rate,
        iterations=1000
    )
    model.fit(X_tr, y_tr)
    lr_scores[rate] = np.mean(model.predict(X_val) == y_val)

best_rate = max(lr_scores, key=lr_scores.get)

lr_model = LogisticRegression(
    learning_rate=best_rate,
    iterations=1000
)
lr_model.fit(X_train, y_train)

# -----------------------------
# 6. SVM
# -----------------------------

svm_scores = {}

for kernel in ['linear', 'rbf']:
    for C in [0.1, 1, 10]:
        model = SVC(kernel=kernel, C=C)
        model.fit(X_tr, y_tr)

        svm_scores[(kernel, C)] = np.mean(
            model.predict(X_val) == y_val
        )

best_kernel, best_C = max(
    svm_scores,
    key=svm_scores.get
)

svm = SVC(kernel=best_kernel, C=best_C, probability=True)
svm.fit(X_train, y_train)

# -----------------------------
# 7. Save trained models
# -----------------------------

model_bundle = {
    'vectorizer': vec,
    'nb_prior': nb.prior,
    'nb_likelihood': nb.likelihood,
    'lr_weights': lr_model.weights,
    'lr_bias': lr_model.bias,
    'svm': svm,
    'remove_words': remove_words,
    'best_alpha': best_alpha,
    'best_rate': best_rate,
    'best_kernel': best_kernel,
    'best_C': best_C
}

joblib.dump(model_bundle, 'model_bundle.pkl')

print("Training completed successfully!")
print("Best NB alpha:", best_alpha)
print("Best LR learning rate:", best_rate)
print("Best SVM:", best_kernel, best_C)
print("Saved trained models as model_bundle.pkl")
