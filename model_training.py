import os

import joblib  # For saving your trained model
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- 1. Load the Pre-processed Data ---
# This assumes your data script saved the clean data to the 'datasets' directory.
PROCESSED_DATA_FILE = "toronto_ksi.csv"  # Use the actual filename
DATASET_DIR = "datasets"

df = pd.read_csv(os.path.join(DATASET_DIR, PROCESSED_DATA_FILE))

print("Clean data loaded successfully!")
print(f"Dataset shape: {df.shape}")


# --- 2. Define Features (X) and Target (y) ---
# Your original script saved 'X' and 'y' separately, but if you saved a single file,
# you'll need to separate the target variable here.
# Assuming 'accident_class' is your target and it's in the loaded CSV.
X = df.drop(columns=["accident_class"])
y = df["accident_class"]


# --- 3. Split Data into Training and Testing Sets ---
# Using stratify is crucial for imbalanced datasets like this one.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


# --- 4. Build Machine Learning Pipelines ---
# Using pipelines is a best practice to prevent data leakage and streamline your workflow.
# A pipeline bundles a scaler (optional but good practice) and a model.

# Pipeline for Logistic Regression
pipeline_lr = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
        ),
    ]
)

# Pipeline for Random Forest
pipeline_rf = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(random_state=42, class_weight="balanced"),
        ),
    ]
)

# Pipeline for Gradient Boosting with scale_pos_weight
# NOTE: GradientBoostingClassifier does not have class_weight.
# We must adjust a parameter inside the model after creating the pipeline.
pipeline_gb = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier(random_state=42)),
    ]
)
# scale_pos_weight = (
#     y_train.value_counts()["Non-Fatal Injury"] / y_train.value_counts()["Fatal"]
# )
# pipeline_gb.set_params(classifier__scale_pos_weight=scale_pos_weight)

# Pipeline extreme trees
pipeline_et = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", ExtraTreesClassifier(random_state=42, class_weight="balanced")),
    ]
)

smote_pipeline = ImbPipeline(
    [
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

pipeline_gb_smote = ImbPipeline(
    [
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier(random_state=42)),
    ]
)


# List of pipelines to iterate through
pipelines = [
    pipeline_lr,
    pipeline_rf,
    pipeline_gb,
    pipeline_et,
    smote_pipeline,
    pipeline_gb_smote,
]


# --- 5. Train and Evaluate Models ---
# Loop through the pipelines to train and evaluate each one.

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for pipe in pipelines:
    print(f"\n===== Training {pipe.steps[-1][1]} =====")

    # Train the model
    pipe.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipe.predict(X_test)

    # Evaluate the model
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0.0))

    print("\nCross-validation scores:")
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# --- 6. Save the Best Performing Model (Example) ---
# After reviewing the evaluation metrics, choose your best model and save it.
# Let's assume Random Forest was the best.
# best_model_pipeline = pipeline_rf
#
# # Create a directory to save models if it doesn't exist
# MODELS_DIR = "models"
# if not os.path.exists(MODELS_DIR):
#     os.makedirs(MODELS_DIR)
#
# # Save the entire pipeline (scaler + model)
# joblib.dump(best_model_pipeline, os.path.join(MODELS_DIR, "best_ksi_model.pkl"))
#
# print("\nBest model saved successfully to 'models/best_ksi_model.pkl'")


# improvements:
# 1. Use StratifiedKFold for cross-validation to maintain class distribution.
# 2. use `class_weight='balanced'` in classifiers to handle class imbalance.automatically adjusts the weights of each class in the model's loss function, penalizing the model more heavily for misclassifying the minority ('Fatal') class
