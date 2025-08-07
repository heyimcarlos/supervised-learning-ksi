import os

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

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

params_grid = {
    "smote__k_neighbors": [3, 5, 7],
    "classifier__n_estimators": range(25, 3001, 25),
    "classifier__max_depth": [5, 8, 10],
    "classifier__min_samples_leaf": [10, 20, 30, 40],
    "classifier__max_features": ["sqrt", "log2"],
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
    n_iter=100,  # Number of parameter settings that are sampled
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
