import os
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, recall_score, precision_recall_curve, precision_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

DATASET_DIR = "datasets"

X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATASET_DIR, "y_train.csv")).squeeze()

# SVM pipeline with scaling (required for SVM) and class balancing
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # SVM requires feature scaling
    ("classifier", SVC(
        random_state=42,
        class_weight='balanced',  # Handles class imbalance
        probability=True  # Needed for threshold optimization
    ))
])

# SVM-specific parameter grid
params_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],  # Regularization parameter
    "classifier__kernel": ['linear', 'rbf', 'poly'],  # Kernel types
    "classifier__gamma": ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # For rbf/poly kernels
    "classifier__degree": [2, 3, 4],  # For poly kernel only
    "classifier__coef0": [0.0, 0.1, 1.0]  # For poly/sigmoid kernels
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
recall_scorer = make_scorer(recall_score, pos_label="Fatal")

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=params_grid,
    scoring=recall_scorer,
    cv=cv_strategy,
    verbose=3,
    random_state=42,
    n_jobs=-1,
    n_iter=50,
)

print("Training SVM model...")
random_search.fit(X_train, y_train)
print("Best parameters found: ", random_search.best_params_)
joblib.dump(random_search.best_estimator_, "models/best_svm.pkl")

# Testing and threshold optimization
X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATASET_DIR, "y_test.csv")).squeeze()

best_model = random_search.best_estimator_

# Standard predictions
y_pred = best_model.predict(X_test)

print("\n" + "=" * 50)
print("STANDARD SVM PREDICTIONS")
print("=" * 50)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Threshold optimization for practical deployment
y_proba = best_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba, pos_label="Fatal")

print("\n" + "=" * 50)
print("THRESHOLD OPTIMIZATION FOR DEPLOYMENT")
print("=" * 50)

# Find optimal thresholds for different recall targets
target_recalls = [0.6, 0.7, 0.8, 0.9]
min_precision = 0.1  # Minimum acceptable precision

deployment_options = []
for target_recall in target_recalls:
    valid_indices = np.where(recalls >= target_recall)[0]

    if len(valid_indices) > 0:
        valid_precisions = precisions[valid_indices]
        best_idx = valid_indices[np.argmax(valid_precisions)]

        if precisions[best_idx] >= min_precision:
            deployment_options.append({
                'target_recall': target_recall,
                'threshold': thresholds[best_idx],
                'actual_recall': recalls[best_idx],
                'precision': precisions[best_idx],
                'f1': 2 * (precisions[best_idx] * recalls[best_idx]) / (precisions[best_idx] + recalls[best_idx])
            })

# Display deployment options
for option in deployment_options:
    print(f"\nTarget {option['target_recall'] * 100:.0f}% Fatal Recall:")
    print(f"  Threshold: {option['threshold']:.3f}")
    print(f"  Actual Recall: {option['actual_recall']:.3f}")
    print(f"  Precision: {option['precision']:.3f}")
    print(f"  F1-Score: {option['f1']:.3f}")

# Choose best balanced option
if deployment_options:
    best_option = max(deployment_options, key=lambda x: x['f1'])
    optimal_threshold = best_option['threshold']

    print(f"\n" + "=" * 50)
    print(f"RECOMMENDED DEPLOYMENT THRESHOLD: {optimal_threshold:.3f}")
    print("=" * 50)

    # Make optimized predictions
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    y_pred_optimal = ["Fatal" if pred == 1 else "Non-Fatal Injury" for pred in y_pred_optimal]

    print("Optimized Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_optimal))
    print("\nOptimized Classification Report:")
    print(classification_report(y_test, y_pred_optimal))

    final_recall = recall_score(y_test, y_pred_optimal, pos_label="Fatal")
    final_precision = precision_score(y_test, y_pred_optimal, pos_label="Fatal")

    print(f"\nFinal Fatal Recall: {final_recall:.3f}")
    print(f"Final Fatal Precision: {final_precision:.3f}")

# SVM-specific analysis: Support vectors info
print("\n" + "=" * 50)
print("SVM MODEL ANALYSIS")
print("=" * 50)
svm_classifier = best_model.named_steps['classifier']
print(f"Number of support vectors: {svm_classifier.n_support_}")
print(f"Support vectors per class: Fatal={svm_classifier.n_support_[0]}, Non-Fatal={svm_classifier.n_support_[1]}")
print(f"Total support vectors: {sum(svm_classifier.n_support_)}")
print(f"Best kernel: {svm_classifier.kernel}")
print(f"Best C parameter: {svm_classifier.C}")
print(f"Best gamma: {svm_classifier.gamma}")