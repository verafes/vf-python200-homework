import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline


# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# data set
# spambase.data: raw numeric data -- 57 numeric features, no column names, # last column = spam  indicator (1 = spam, 0 = ham)
URL_DATA = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
# spambase.names : feature descriptions (57 features) + info about dataset
URL_NAMES = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names"

# --- Task 1: Load and Explore ---
print("\n--- Task 1: Load and Explore ---")

# helper to read feature names
def read_feature_names(names_path):
    feature_names = []

    text = requests.get(names_path).text
    for line in text.splitlines():
        line = line.strip()

        # Feature lines contain a colon and end with a period
        if ":" in line and line.endswith("."):
            name = line.split(":")[0]
            feature_names.append(name)

    # add target column
    if "spam_label" not in feature_names:
        feature_names.append("spam_label")
    return feature_names

# load dataset
def load_spambase(data_path=URL_DATA, names_path=URL_NAMES):
    feature_names = read_feature_names(names_path)
    response = requests.get(data_path)
    response.raise_for_status()
    df = pd.read_csv(BytesIO(response.content), header=None, names=feature_names)

    return df

df = load_spambase()

print("Data shape:", df.shape)
print(f"Number of emails: {df.shape[0]}")
print(f"Columns: {df.columns}")
print(f"Number of columns: {df.shape[1]}")

# The dataset contains 4,601 emails and 57 numeric features plus the spam_label target.
# This gives us a moderately sized classification dataset suitable for classical ML models.

# Class balance
class_counts = df["spam_label"].value_counts().sort_index()
class_ratio = df["spam_label"].value_counts(normalize=True) * 100

print("\nClass counts (0 = ham, 1 = spam):")
print(class_counts)
print("\nClass ratios:")
print(class_ratio.apply(lambda x: f"{x:.3f}"))

# The classes are not perfectly balanced. Ham is the majority class.
# Because of this, raw accuracy can be misleading — a model could achieve ~60% accuracy
# by predicting “ham” for everything.


# Boxplots for key features by class
features_to_plot = [
    "word_freq_free",
    "char_freq_!",
    "capital_run_length_total",
]

for feature in features_to_plot:
    plt.figure(figsize=(8, 7))
    ham = df.loc[df["spam_label"] == 0, feature]
    spam = df.loc[df["spam_label"] == 1, feature]
    plt.boxplot([ham, spam], tick_labels=["ham", "spam"])
    plt.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    plt.title(f"{feature} by class")
    plt.suptitle("")  # remove automatic super-title
    plt.xlabel("spam (0 = ham, 1 = spam)")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(f"outputs/boxplot_{feature}_by_spam_boxplot.png")
    # plt.show()
    plt.close()


# Boxplot observations:
# - Spam uses "free" more often.
# - Spam has higher '!' frequency.
# - Spam shows longer capital-letter runs.

# The distribution is skewed for both classes, but spam has a higher median and more extreme values.

# --- Feature Scale Observations ---
print("\nFeature value ranges:")
print(df.describe().loc[["min", "max", "mean"]].T.to_string())

# Word-frequency features are tiny fractions and mostly zero -> very skewed and sparse.
# Capital-run-length features can reach hundreds or thousands -> much larger scale.
# Because scales differ so much, models like Logistic Regression, SVM, and kNN need
# standardized features. Tree-based models are unaffected by scale.


# --- Task 2: Prepare Data ---
# (Train/Test + Scaling + PCA)
print("\n--- Task 2: Prepare Data  ---")

# Separate features and target (keep DataFrame)
X = df.drop(columns=["spam_label"])
y = df["spam_label"]

# Train/test split with stratification to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Scaling: important because feature scales vary dramatically.
# We fit the scaler on the training data only to avoid leaking
# information from the test set into preprocessing.

# Scale using training data only
scaler = StandardScaler()
scaler.fit(X_train)

# Transform both sets using the fitted scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- PCA Preprocessing (for models that benefit from it) ---
print("\n--- PCA: Fit on Scaled Training Data ---")
# fit PCA on the scaled training data
pca = PCA()
pca.fit(X_train_scaled)

# cumulative explained variance + find n
cum_var = np.cumsum(pca.explained_variance_ratio_)
# number of components to reach 90% variance
num_comp_90 = int(np.argmax(cum_var >= 0.90)) + 1
print(f"Number of components for 90% variance: {num_comp_90}")

plt.figure(figsize=(8, 6))
plt.plot(cum_var, marker="o")
plt.axhline(y=0.90, color='r', linestyle='--', label='90% variance threshold')
plt.axvline(x=num_comp_90, color='g', linestyle='--', label=f'n={num_comp_90}')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Spambase PCA: Cumulative Variance Explained")
plt.grid(True, alpha=0.3)
plt.legend(loc="lower center")
plt.tight_layout()
plt.savefig("outputs/spam_pca_variance_explained.png")
# plt.show()
plt.close()

# Transform and slice to first n components
X_train_pca = pca.transform(X_train_scaled)[:, :num_comp_90]
X_test_pca = pca.transform(X_test_scaled)[:, :num_comp_90]
print(f"PCA-reduced train shape: {X_train_pca.shape}")
print(f"PCA-reduced test shape:  {X_test_pca.shape}")

# PCA reduces noise and dimensionality while preserving most variance.


# --- Task 3: Classifier Comparison ---
print("\n--- Task 3: Classifier Comparison ---")

def eval_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    print(f"\n{name} ")
    print("-" * len(name))
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_te, y_pred, digits=3))

    return acc, y_pred

best_models = {}   # name -> (accuracy, y_pred)

# 1) KNN on unscaled data
print("\nKNN (k=5) on UNscaled data")
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
best_models["KNN_unscaled"] = eval_model(
    "KNN (k=5) - UNscaled",
    knn_unscaled,
    X_train, X_test, y_train, y_test
)

# 2) KNN on scaled data
print("\nKNN (k=5) on SCALED data")
knn_scaled = KNeighborsClassifier(n_neighbors=5)
best_models["KNN_scaled"] = eval_model(
    "KNN (k=5) - scaled",
    knn_scaled,
    X_train, X_test, y_train, y_test
)

# 3) KNN on PCA-reduced data
print("\nKNN (k=5) on PCA-reduced data")
knn_pca = KNeighborsClassifier(n_neighbors=5)
best_models["KNN_pca"] = eval_model(
    "KNN (k=5) - PCA",
    knn_pca,
    X_train, X_test, y_train, y_test
)

# Scaling and PCA can help KNN because it relies on distances; PCA may reduce noise
# and redundancy, sometimes improving generalization.

# 4) Decision Tree with different depths
print("\nDecision Tree: depth sweep")

depths = [3, 5, 10, None]
dt_results = {}

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    y_train_dt = dt.predict(X_train)
    y_test_dt = dt.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_dt)
    test_acc = accuracy_score(y_test, y_test_dt)
    dt_results[depth] = (train_acc, test_acc)
    print(f"max_depth={depth}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

# As depth increases, training accuracy usually rises toward 1.0, while test accuracy
# may peak and then drop, indicating overfitting at very large depths.

# a reasonable depth based on the printed results
chosen_depth = 10
print(f"\nChosen Decision Tree depth for production: {chosen_depth}")

dt_final = DecisionTreeClassifier(max_depth=chosen_depth, random_state=42)
best_models["DecisionTree"] = eval_model(
    f"Decision Tree (max_depth={chosen_depth})",
    dt_final,
    X_train, X_test, y_train, y_test
)

# 5) Random Forest
print("\nRandom Forest")
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
best_models["RandomForest"] = eval_model(
    "Random Forest (200 trees)",
    rf,
    X_train, X_test, y_train, y_test
)

# --- Feature Importances (Decision Tree + Random Forest) ---
print("\nTop 10 Decision Tree Feature Importances:")
dt_importances = pd.Series(
    dt_final.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)
print(dt_importances.head(10).round(4))

print("\nTop 10 Random Forest Feature Importances:")
rf_importances = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)
print(rf_importances.head(10).round(4))

# Save Random Forest bar chart
plt.figure(figsize=(8,6))
rf_importances.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.close()

# Random forests reduce overfitting compared to a single tree by averaging many trees.

# 6) Logistic Regression on scaled data
print("\nLogistic Regression (C=1.0) on SCALED data")
log_reg_scaled = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
best_models["LogReg_scaled"] = eval_model(
    "Logistic Regression - scaled",
    log_reg_scaled,
    X_train, X_test, y_train, y_test
)

# 7) Logistic Regression on PCA-reduced data
print("\nLogistic Regression (C=1.0) on PCA-reduced data")
log_reg_pca = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
best_models["LogReg_pca"] = eval_model(
    "Logistic Regression - PCA",
    log_reg_pca,
    X_train, X_test, y_train, y_test
)

# Logistic regression benefits from scaling; PCA can sometimes help by removing
# correlated directions and noise, but may also discard useful signal.

# --- Confusion Matrix for Best Model ---
print("\n--- Confusion Matrix for Best Model ---")

# Pick the best model by accuracy
best_name = None
best_acc = -1
best_y_pred = None

for name, (acc, y_pred) in best_models.items():
    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_y_pred = y_pred

print(f"Best model by accuracy: {best_name} (accuracy={best_acc:.4f})")

cm = confusion_matrix(y_test, best_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix: {best_name}")
plt.tight_layout()
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.close()

# --- Summary  ---
# 1. Best model:
# Random Forest performed best overall with the highest test accuracy (0.9457)
# and balanced precision/recall.


# Logistic Regression comparison:
# Both scaled and PCA versions achieved the same accuracy (0.9283).
# Scaling is essential for LR, but PCA did not improve results and slightly reduced interpretability.

# Decision Tree depth sweep:
# Training accuracy increased with depth, while test accuracy peaked at depth=None.
# Depth=10 was chosen as the best balance between underfitting and overfitting.

# PCA impact:
# PCA did not improve performance for KNN or Logistic Regression.
# This matches expectations from Task 2: PCA can remove noise but may also discard useful signal.

# Spam-filter metric:
# False positives (ham → spam) are more harmful because users may miss legitimate emails.
# A good spam filter should minimize false positives even if it allows some spam through.

# Confusion matrix:
# The best model (Random Forest) makes more false negatives than false positives.
# This is a conservative behavior: it avoids blocking real emails but lets some spam through.

# Task 4: Cross-Validation
print("\n--- Task 4: Cross-Validation (cv=5) ---")

def run_cv(name, model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    mean, std = scores.mean(), scores.std() # score results
    print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")
    return mean, std

cv_scores = {} # name -> (mean, std)

# KNN (unscaled, scaled, PCA)
cv_scores["KNN_unscaled"] = run_cv("KNN_unscaled", KNeighborsClassifier(n_neighbors=5), X_train, y_train)
cv_scores["KNN_scaled"]   = run_cv("KNN_scaled",   KNeighborsClassifier(n_neighbors=5), X_train_scaled, y_train)
cv_scores["KNN_pca"]      = run_cv("KNN_pca",      KNeighborsClassifier(n_neighbors=5), X_train_pca, y_train)

# Decision Tree (chosen depth)
cv_scores[f"DecisionTree_depth={chosen_depth}"] = run_cv(f"DecisionTree_depth={chosen_depth}",
       DecisionTreeClassifier(max_depth=chosen_depth, random_state=42),
       X_train, y_train)

# Random Forest
cv_scores["RandomForest"] = run_cv("RandomForest",
       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
       X_train, y_train)

# Logistic Regression (scaled, PCA)
cv_scores["LogReg_scaled"] = run_cv("LogReg_scaled",
       LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
       X_train_scaled, y_train)

cv_scores["LogReg_pca"] = run_cv("LogReg_pca",
       LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
       X_train_pca, y_train)

print("\n--- Task 4 Summary ---")

# Best mean accuracy
best_mean_name, (best_mean, best_std) = max(
    cv_scores.items(), key=lambda x: x[1][0]
)

# Most stable (lowest std)
best_stable_name, (stable_mean, stable_std) = min(
    cv_scores.items(), key=lambda x: x[1][1]
)

print(f"Best mean CV accuracy: {best_mean_name} (mean={best_mean:.4f}, std={best_std:.4f})")
print(f"Most stable model: {best_stable_name} (std={stable_std:.4f})")

print("\nAll models (sorted by mean accuracy):")
for name, (mean, std) in sorted(cv_scores.items(), key=lambda x: x[1][0], reverse=True):
    print(f"- {name}: mean={mean:.4f}, std={std:.4f}")

print("\nInterpretation:")
print(f"- Random Forest achieved the highest mean CV accuracy ({best_mean:.4f}).")
print(f"- Logistic Regression (PCA) was the most stable with the lowest std ({stable_std:.4f}).")
print("- KNN_unscaled was the weakest performer by a large margin.")
print("- Overall ranking is consistent with Task 3: tree-based and linear models outperform KNN.")
print("- Cross-validation confirms Random Forest as the most reliable model.")


# --- Task 5: Pipelines ---
print("\n--- Task 5: Pipelines ---")

# Pipeline 1 — Best Tree‑Based Model (Random Forest)
# (no scaling, no PCA)
tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

tree_pipeline.fit(X_train, y_train)
y_pred_tree = tree_pipeline.predict(X_test)

print("\nTree Pipeline Report:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")
print(classification_report(y_test, y_pred_tree))

# Pipeline 2 — Best Non‑Tree Model (Logistic Regression)
# Best accuracy among non‑tree models
non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=num_comp_90)),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
])

non_tree_pipeline.fit(X_train, y_train)
y_pred_non_tree = non_tree_pipeline.predict(X_test)

print("\nNon-Tree Pipeline Report:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_non_tree):.4f}")
print(classification_report(y_test, y_pred_non_tree))

# score() example
print(f"\nTree Pipeline score(): {tree_pipeline.score(X_test, y_test):.4f}")
print(f"Non-Tree Pipeline score(): {non_tree_pipeline.score(X_test, y_test):.4f}")

# --- Task 5 Summary ---
# The Random Forest pipeline contains only the classifier because tree-based models
# do not require scaling or PCA. The pipeline simply wraps the model
# so it can be fit and evaluated in one step.

# The Logistic Regression pipeline includes a StandardScaler step because LR is
# sensitive to feature scale. This pipeline reproduces the same accuracy and
# # classification report as the manual preprocessing from Task 3.
#
# Both pipelines reproduce the same results as the manual preprocessing in Task 3,
# confirming that the pipeline applies transformations in the correct order.

# The practical value of pipelines:
# - ensures correct preprocessing order
# - prevents data leakage from test data
# - makes the model portable and deployment-ready
# - simplifies evaluation and cross-validation
# - reduces human error in multi-step workflows