import os

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

DATASET_DIR = "datasets"

X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train.csv"))
y_train = pd.read_csv(
    os.path.join(DATASET_DIR, "y_train.csv")
).squeeze()  # converts DataFrame to Series


pipeline = Pipeline(
    [
        ("smote", SMOTE(random_state=42)),
        (
            "classifier",
            RandomForestClassifier(random_state=42),
        ),
    ]
)

#  WARNING: runs:
# 1: Best Params: {'smote__k_neighbors': 3, 'classifier__n_estimators': 1550, 'classifier__min_samples_leaf': 20, 'classifier__max_features': 'sqrt', 'classifier__max_depth': 2}
# 2: Best Params:  {'smote__k_neighbors': 7, 'classifier__n_estimators': 25, 'classifier__min_samples_leaf': 2, 'classifier__max_features': 'sqrt', 'classifier__max_depth': 1}
# 3: Best parameters found:  {'smote__k_neighbors': 5, 'classifier__n_estimators': 325, 'classifier__min_samples_leaf': 20, 'classifier__max_features': 'sqrt', 'classifier__max_depth': 2}
params_grid = {
    "smote__k_neighbors": [5, 7, 10, 15],
    "classifier__n_estimators": range(25, 3001, 25),
    "classifier__max_depth": [2, 4, 5],
    "classifier__min_samples_leaf": [2, 5, 10, 20, 30],
    "classifier__max_features": ["sqrt"],
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#  NOTE: Use recall as the scoring metric because the dataset is imbalanced
recall_scorer = make_scorer(recall_score, pos_label="Fatal")

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=params_grid,
    scoring=recall_scorer,
    cv=cv_strategy,
    verbose=3,
    random_state=42,
    n_jobs=-1,
    n_iter=50,  # Number of parameter settings that are sampled
)
random_search.fit(X_train, y_train)
print("Best parameters found: ", random_search.best_params_)
joblib.dump(random_search.best_estimator_, "models/best_random_forest.pkl")


# grid_search = GridSearchCV(
#     estimator=pipeline,
#     param_grid=params_grid,
#     scoring=recall_scorer,
#     cv=cv_strategy,
#     verbose=3,
#     n_jobs=-1,
# )
# grid_search.fit(X_train, y_train)
#
# print("Best parameters found: ", grid_search.best_params_)
# joblib.dump(grid_search.best_estimator_, "best_random_forest_model.pkl")

# Testing the best model

# Load test data
X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATASET_DIR, "y_test.csv")).squeeze()

# Get the best model from random search
best_model = random_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Print evaluation metrics
print("\n" + "=" * 50)
print("Best Random Forest Model Evaluation")
print("=" * 50)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

fatal_recall = recall_score(y_test, y_pred, pos_label="Fatal")
print(f"\nFatal Class Recall: {fatal_recall:.4f}")
print(
    f"This means the model correctly identifies {fatal_recall*100:.2f}% of all fatal accidents"
)
