import os
import re

import joblib
import lightgbm as lgb
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

DATASET_DIR = "datasets"

X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATASET_DIR, "y_train.csv")).squeeze()
X_train.columns = [re.sub(r"[^A-Za-z0-9_]+", "", col) for col in X_train.columns]

pipeline = Pipeline(
    [
        ("smote", SMOTE(random_state=42)),
        (
            "classifier",
            lgb.LGBMClassifier(
                random_state=42,
                class_weight="balanced",  # Handles class imbalance
            ),
        ),
    ]
)


# 1: Best parameters found:  {'classifier__reg_lambda': 1.0, 'classifier__reg_alpha': 0.5, 'classifier__num_leaves': 31, 'classifier__n_estimators': 100, 'classifier__min_child_samples': 70, 'classifier__max_depth': 5, 'classifier__learning_rate': 0.01}
param_grid = {
    # Relationship between learning rate and number of trees
    "classifier__n_estimators": [100, 200, 500],
    # "classifier__learning_rate": [0.01, 0.05, 0.1],
    # Core parameters for controlling tree complexity
    "classifier__num_leaves": [20, 30, 40, 255],
    "classifier__max_depth": [5, 8, 10],  #
    # Parameters for preventing overfitting (using correct names)
    "classifier__min_child_samples": [30, 50, 70, 100],
    # "classifier__reg_alpha": [0, 0.1, 0.5, 1.0],  # Corrected L1 regularization
    # "classifier__reg_lambda": [0, 0.1, 0.5, 1.0],  # Corrected L2 regularization
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
recall_scorer = make_scorer(recall_score, pos_label="Fatal")

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=50,
    scoring=recall_scorer,
    cv=cv_strategy,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

# --- 5. Run the Search ---
random_search.fit(X_train, y_train)

# --- 6. Save and Evaluate the Best Model ---
print("Best parameters found: ", random_search.best_params_)
joblib.dump(random_search.best_estimator_, "models/best_lightgbm.pkl")

print("\n--- Evaluating Best Tuned LightGBM Model on Test Set ---")
X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATASET_DIR, "y_test.csv")).squeeze()
best_lgbm = random_search.best_estimator_
y_pred = best_lgbm.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
