import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

# --- Define Directories ---
DATASET_DIR = "datasets"
MODELS_DIR = "models"

# --- Load the training data AND your best-tuned Random Forest pipeline ---
# We need X_train here to get the correct column names that the model was trained on.
X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATASET_DIR, "y_train.csv")).squeeze()
X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test.csv"))

best_rf_pipeline = joblib.load(os.path.join(MODELS_DIR, "best_random_forest.pkl"))

rf_model = best_rf_pipeline.named_steps["classifier"]
importances = rf_model.feature_importances_
feature_names = X_train.columns

# --- CRUCIAL DEBUGGING STEP ---
# Print the lengths to see if they match.
# print(f"Number of feature importances: {len(importances)}")
# print(f"Number of feature names: {len(feature_names)}")
#
#
# # --- Check for mismatch and proceed only if lengths are equal ---
# if len(importances) == len(feature_names):
#     # Create a DataFrame for better visualization
#     feature_importance_df = pd.DataFrame(
#         {"feature": feature_names, "importance": importances}
#     ).sort_values(by="importance", ascending=False)
#
#     # --- Display the top 20 most important features ---
#     print("\nTop 20 Most Important Features:")
#     print(feature_importance_df.head(20))
#
#     # --- Visualize the feature importances ---
#     plt.figure(figsize=(10, 8))
#     plt.barh(
#         feature_importance_df["feature"][:20], feature_importance_df["importance"][:20]
#     )
#     plt.xlabel("Feature Importance")
#     plt.ylabel("Feature")
#     plt.title("Top 20 Features from Tuned Random Forest")
#     plt.gca().invert_yaxis()
#     plt.tight_layout()
#     plt.show()
# else:
#     print(
#         "\nERROR: Mismatch found between the number of features in the trained model and the provided column names."
#     )
#     print(
#         "Please ensure that the 'X_train.csv' file is the exact same version that the model was trained on."
#     )

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import RFECV
# from sklearn.model_selection import StratifiedKFold
#
# # --- Use a simpler RandomForestClassifier for this, as it will be trained many times ---
# # NOTE: Do not use your full SMOTE pipeline here, as RFECV is very slow.
# # This is an investigation, not the final model training.
# estimator = RandomForestClassifier(
#     random_state=42, class_weight="balanced", n_estimators=50
# )
#
# # --- Set up the RFECV selector ---
# # It will use stratified k-fold cross-validation to find the best number of features
# cv_strategy = StratifiedKFold(n_splits=3)  # Use fewer splits to speed it up
# rfecv = RFECV(
#     estimator=estimator,
#     step=1,  # Remove one feature at a time
#     cv=cv_strategy,
#     scoring="recall_macro",  # Optimize for recall
#     n_jobs=-1,
# )
#
# print("Running RFECV... this may take a very long time.")
# rfecv.fit(X_train, y_train)
#
# print(f"Optimal number of features: {rfecv.n_features_}")
#
# # --- Get the list of the best features ---
# selected_features = X_train.columns[rfecv.support_]
# print("Selected Features:")
# print(selected_features)
#
# # Assuming 'rfecv' is your fitted RFECV object and X_train_full/X_test_full are your original data
# selected_features = X_train.columns[rfecv.support_]
#
# # Create the new DataFrames
# X_train_selected = X_train[selected_features]
# X_test_selected = X_test[selected_features]
#
# # Save them for your other scripts
# X_train_selected.to_csv("datasets/X_train_selected.csv", index=False)
# X_test_selected.to_csv("datasets/X_test_selected.csv", index=False)
#

# --- Load your best-tuned model and test data ---
best_lr_model = joblib.load("models/best_logistic_regression.pkl")
X_test = pd.read_csv("datasets/X_test.csv")
y_test = pd.read_csv("datasets/y_test.csv").squeeze()

# --- 1. Get the prediction PROBABILITIES for the 'Fatal' class ---
# We need probabilities, not the final 0/1 predictions.
# The second column [:, 1] corresponds to the positive class ('Fatal').
y_pred_probs = best_lr_model.predict_proba(X_test)[:, 1]
print(
    f"Predicted probabilities for 'Fatal' class: {y_pred_probs[:5]}"
)  # Show first 5 for debugging

# --- 2. Calculate precision, recall, and thresholds ---
precision, recall, thresholds = precision_recall_curve(
    y_test, y_pred_probs, pos_label="Fatal"
)

# --- 3. Find the best threshold that maximizes F1-score ---
# We add a small epsilon to avoid division by zero
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

# Find the threshold that corresponds to the best F1 score
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_index]
best_f1 = f1_scores[best_threshold_index]

print(f"Best Threshold found: {best_threshold:.4f}")
print(f"This gives a balanced F1-Score of: {best_f1:.4f}")

# --- 4. Apply the new threshold to get your final predictions ---
y_pred_optimized = (y_pred_probs >= best_threshold).astype(int)
# We need to convert y_test to int for comparison
y_test_int = (y_test == "Fatal").astype(int)

print("\n--- Performance with Optimized Threshold ---")

print(
    classification_report(
        y_test_int,
        y_pred_optimized,
        target_names=["Non-Fatal Injury", "Fatal"],
        zero_division=0.0,
    )
)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_int, y_pred_optimized))
