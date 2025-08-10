import os

import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET_DIR = "datasets"

X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test.csv"))

y_train = pd.read_csv(
    os.path.join(DATASET_DIR, "y_train.csv")
).squeeze()  # converts DataFrame to Series
y_test = pd.read_csv(
    os.path.join(DATASET_DIR, "y_test.csv")
).squeeze()  # converts DataFrame to Series

model_rf = joblib.load("models/best_random_forest.pkl")
model_lr = joblib.load("models/best_logistic_regression.pkl")
# model_gb = joblib.load("models/best_gradient_boosting.pkl")
# model_svm = joblib.load("models/best_svm.pkl")

estimators = [
    # ("random_forest", model_rf),
    ("logistic_regression", model_lr),
    # ("svm", model_svm),
    # ("gradient_boosting", model_gb),
]

voting_clf = VotingClassifier(
    estimators=estimators,
    voting="soft",  # Use the probabilities from your fine-tuned models
    # weights=[0.3, 0.7],
)

# --- Fit and Evaluate the Final Ensemble ---
# You fit this on the full training data one last time
voting_clf.fit(X_train, y_train)

# Evaluate on the test set
y_pred = voting_clf.predict(X_test)

print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

print("Final Voting Classifier Performance:")
print(classification_report(y_test, y_pred))

print("confusion matrix:")
print(confusion_matrix(y_test, y_pred))


# Save the final voting classifier model
joblib.dump(voting_clf, "models/voting_classifier.pkl")
