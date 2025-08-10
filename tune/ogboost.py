import os
import re

import joblib
import pandas as pd
import xgboost as xgb

# from ogboost import OGBoost
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

DATASET_DIR = "datasets"

X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train_selected.csv"))
y_train_raw = pd.read_csv(os.path.join(DATASET_DIR, "y_train.csv")).squeeze()


X_train.columns = [re.sub(r"[^A-Za-z0-9_]+", "", col) for col in X_train.columns]

le = LabelEncoder()

y_train = le.fit_transform(y_train_raw)

scale_pos_weight = y_train_raw.value_counts()[0] / y_train_raw.value_counts()[1]

pipeline = Pipeline(
    [
        (
            "classifier",
            xgb.XGBClassifier(
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
            ),
        )
    ]
)

# m = xgb.XGBClassifier()
# m.eval_metric

# Best parameters found:  {'classifier__subsample': 1.0, 'classifier__n_estimators': 100, 'classifier__max_depth': 3, 'classifier__learning_rate': 0.05, 'classifier__colsample_bytree': 0.8}
param_grid = {
    "classifier__n_estimators": [100, 200, 500, 1000],
    "classifier__max_depth": [3, 5, 7, 10],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],  # Ratio of training instances
    "classifier__colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0,
    ],  # Ratio of columns when constructing each tree
}


# --- 5. Set up and Run RandomizedSearchCV ---
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
recall_scorer = make_scorer(recall_score, pos_label=1)

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

random_search.fit(X_train, y_train)


# --- 6. Save and Evaluate the Best Model ---
print("Best parameters found: ", random_search.best_params_)
joblib.dump(random_search.best_estimator_, "models/best_xgboost.pkl")

print("\n--- Evaluating Best Tuned XGBoost Model on Test Set ---")
X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test_selected.csv"))
y_test_raw = pd.read_csv(os.path.join(DATASET_DIR, "y_test.csv")).squeeze()
X_test.columns = [re.sub(r"[^A-Za-z0-9_]+", "", col) for col in X_test.columns]


y_test = le.fit_transform(y_test_raw)

best_xgb = random_search.best_estimator_

y_pred = best_xgb.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
