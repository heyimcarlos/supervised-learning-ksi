import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_DIR = "datasets"

X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train.csv"))
y_train = pd.read_csv(
    os.path.join(DATASET_DIR, "y_train.csv")
).squeeze()  # converts DataFrame to Series


pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(random_state=42, max_iter=2500, class_weight="balanced"),
        ),
    ]
)

# adding the L1 penalty is a great idea because it provides a form of automatic feature selection.
# L2 Penalty (Ridge): Shrinks the coefficients of less important features closer to zero, but rarely to exactly zero.
# L1 Penalty (Lasso): Can shrink the coefficients of unimportant features to exactly zero.
params_grid = [
    {
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs", "liblinear"],
    },
    {
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1"],
        "classifier__solver": ["saga"],
    },
]
#  NOTE: BEST
# 'classifier__C': 0.001: This is the most important result. A very small C value indicates that strong regularization is needed for your model to perform best. This helps prevent overfitting by penalizing large coefficient values, effectively creating a simpler, more generalized model.
# 'classifier__penalty': 'l2': The grid search confirmed that the L2 (Ridge) penalty is the best choice.
# 'classifier__solver': 'liblinear': For this combination of data and parameters, 'liblinear' was the most effective optimization algorithm.

cv_strategy = StratifiedKFold(n_splits=12, shuffle=True, random_state=42)

# NOTE: Use recall as the scoring metric because the dataset is imbalanced
recall_scorer = make_scorer(recall_score, pos_label="Fatal")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    scoring=recall_scorer,
    cv=cv_strategy,
    verbose=3,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
joblib.dump(
    grid_search.best_estimator_,
    "models/best_logistic_regression.pkl",
)

# Look at the recall for the "Fatal" class: 0.65. This is a massive improvement over your previous models. It means you are now correctly identifying 65% of all fatal accidents, up from less than 20% before.

# This happened because class_weight="balanced" worked perfectly. It forced the model to pay much more attention to the minority 'Fatal' class. The trade-off is that its precision dropped to 0.28 (meaning more false positives), but for a problem like this, high recall on the 'Fatal' class is almost always the most important goal.
