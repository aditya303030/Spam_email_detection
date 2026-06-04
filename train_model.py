import pandas as pd
import joblib
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Training with sklearn version:", sklearn.__version__)

# Load data
df = pd.read_csv("data/spam.csv")

# Remove duplicate messages
df_clean = df.drop_duplicates(subset=["Message"]).copy()

X = df_clean["Message"]
y = df_clean["Category"]

# Train/test split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build pipeline
spam_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("model", LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    ))
])

# Train
spam_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = spam_pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(spam_pipeline, "spam_model.pkl")

print("Model saved to spam_model.pkl")