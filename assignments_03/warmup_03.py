import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Load Iris dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---

# --- Q1: Train/Test Split ---
print("\n--- Preprocessing Q1 ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

# --- Q2: StandardScaler ---
print("\n--- Preprocessing Q2 ---")
scaler = StandardScaler()

# computing mean & std from training data to fit only on training data
scaler.fit(X_train)

# applying those values to scale the data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nColumn means of X_train_scaled (should be ~0):")
print(np.mean(X_train_scaled, axis=0))

# Why fit on X_train only?
# Fitting only on X_train prevents information from the test set leaking into the model.


# --- KNN ---

# Q1: Unscaled Data
print("\n--- KNN Q1 ---")
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)

y_pred_unscaled  = knn_unscaled.predict(X_test)
knn_unscaled_acc = accuracy_score(y_test, y_pred_unscaled)

print(f"Accuracy (unscaled): {knn_unscaled_acc}")
print(f"\nClassification Report (unscaled): ")
print(classification_report(y_test, y_pred_unscaled, target_names=iris.target_names))


# Q2: Unscaled KNN
print("\n--- KNN Q2 ---")

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = knn_scaled.predict(X_test_scaled)

print(f"Accuracy (scaled): {accuracy_score(y_test, y_pred_scaled):.4f}")

# Scaling may improve or make little difference.
# Iris features (cm) are already similar in size, so impact is usually very minor.
# But KNN is distance-based, so scaling is generally a best practice
# to prevent larger units from dominating the distance calculation.


# Q3: Cross-validation for k=5 on unscaled data
print("\n--- KNN Q3 ---")
# creating a new KNeighborsClassifier object
knn_model = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn_model, X_train, y_train, cv=5)

print("Fold scores:")
for s in cv_scores:
    print(f"  {s:.4f}")
print(f"Mean: {cv_scores.mean()}")
print(f"Std Dev: {cv_scores.std():.4f}")

# Cross-validation is more trustworthy because it tests multiple splits,
# not just one train/test split, and reduce the impact of a "lucky" or "unlucky" random split.


# Q4: CV over multiple k values
print("\n--- KNN Q4 ---")

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

best_k = None
best_score = 0 # accuracy never goes below 0

for k in k_values:
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_cv, X_train, y_train, cv=5)
    mean_score = scores.mean()

    print(f"k={k}, Mean CV Score={mean_score:.4f}")

    if mean_score >= best_score: # using >= to pick the LATEST (larger) k in a tie
        best_score = mean_score
        best_k = k

print(f"\nBest k = {best_k} with score = {best_score:.4f}")

# Choosing the k with highest CV score because it performs best across multiple validation folds.
# Smaller k can overfit; larger k can underfit.
# CV helps identify the best balance for this dataset.
# I would choose k=7. While both k=5 and k=7 share the highest mean CV score (0.9750),
# using a slightly larger k makes the model less sensitive to noise.
# This usually helps reduce the risk of overfitting to specific training points
# and gives a more stable result on new data.


# --- Classifier Evaluation ---

# Q1: Confusion Matrix for KNN (unscaled)
print("\n--- Classifier Evaluation Q1 ---")

cm = confusion_matrix(y_test, y_pred_unscaled)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot(cmap="Blues")

plt.title("KNN Confusion Matrix (Unscaled Data)")
plt.tight_layout()
plt.savefig("outputs/knn_confusion_matrix.png")
plt.close()
print(f"\nKNN confusion matrix saved to assignments_03/outputs")

# The confusion matrix shows which species the model mixes up.
# KNN most often confuses Versicolor and Virginica, since they are the most similar.


# --- The sklearn API: Decision Trees ---

print("\n--- Decision Trees Q1 ---")

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

dt_acc = accuracy_score(y_test, y_pred_dt)

print(f"Decision Tree Accuracy: {dt_acc:.4f}")
print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))

# Compared to KNN, the Decision Tree may perform similarly or slightly worse.
# A shallow tree (max_depth=3) can underfit, while KNN adapts more flexibly.
# Decision Trees do not rely on distances, so scaling the features does not
# change the splits or the predictions. Scaled vs. unscaled data gives the same result.


# --- Logistic Regression and Regularization ---

# Q1: Effect of C on Coefficients
print("\n--- Logistic Regression Q1 ---")

C_values = [0.01, 1.0, 100]

for C in C_values:
    log_reg = LogisticRegression(
        C=C, max_iter=1000, solver='liblinear'
    )
    log_reg.fit(X_train_scaled, y_train)

    coef_sum = np.abs(log_reg.coef_).sum()
    print(f"C={C}: Total |coefficients| = {coef_sum:.4f}")

# As C increases, the total size of the coefficients becomes larger.
# This is because C is the inverse of regularization strength;
# A small C means strong regularization, which keeps weights small to prevent overfitting.
# A large C weakens regularization, allowing the model to use larger weights to fit the data.
# With liblinear, this effect is more pronounced because the solver applies stronger
# coefficient shrinkage at small C values, so the total |coefficients| grows more clearly
# as C increases.

# --- PCA (Digits dataset) ---

# Digits dataset : 1797 images of handwritten digits
# each digit is an 8×8 image (64 pixels), each pixel has a value 0–16 (brightness)

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# PCA Q1: Shapes and sample digit grid
print("\n--- PCA Q1 ---")
# sanity check: print to confirm the structure of the dataset
print(f"X_digits shape: {X_digits.shape}")
print(f"Images shape: {images.shape}")

#
plt.figure(figsize=(10, 2))
unique_digits = np.unique(y_digits)

# create a 1-row subplot with 10 images, one example of each digit 0–9
for idx, digit in enumerate(unique_digits):
    ax = plt.subplot(1, 10, idx + 1)
    sample_idx = np.where(y_digits == digit)[0][0]
    ax.imshow(images[sample_idx], cmap='gray_r')
    ax.set_title(str(digit))
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_digits.png")
plt.close()
print(f"\nSample Digits plot saved to assignments_03/outputs")


# PCA Q2: Fit PCA and 2D projection
print("\n--- PCA Q2 ---")

# fit PCA on the full 64‑dimensional digit data
pca = PCA()
pca.fit(X_digits)

# transform the data into PCA 'scores'
scores = pca.transform(X_digits)

# Plot the first two principal components (PC1 vs. PC2)
plt.figure(figsize=(6, 5))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Projection of Handwritten Digits")
plt.colorbar(scatter, label='Digit')
plt.tight_layout()
plt.savefig("outputs/pca_2d_projection.png")
plt.close()
print(f"PCA 2D Projection scatter plot saved to assignments_03/outputs")

# In this 2D PCA space, images of the same digit tend to cluster together,
# although there is still some overlap between similar digits.


# PCA Q3: Cumulative explained variance
print("\n--- PCA Q3 ---")
# Compute the cumulative explained variance
cum_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot it against the number of components (1 through 64)
plt.figure(figsize=(6, 4))
plt.plot(cum_variance, marker='o', color="orange")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: Cumulative Variance Explained")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/pca_variance_explained.png")
plt.close()
print(f"PCA Variance Explained plot saved to assignments_03/outputs")

# Approximate number of components needed for 80% variance
n_80 = np.argmax(cum_variance >= 0.80) + 1
print(f"\nApprox. number of components for 80% variance: {n_80}")

# We need roughly 13 components to explain about 80% of the variance.
# This is where the curve starts to level off and additional components
# contribute less additional information.

# PCA Q4: Reconstructions with different numbers of components
print("\n--- PCA Q4 ---")

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_values = [2, 5, 15, 40]
num_samples = 5  # first 5 digits

plt.figure(figsize=(10, 8))

# Row 0: original images
for j in range(num_samples):
    ax = plt.subplot(len(n_values) + 1, num_samples, j + 1)
    ax.imshow(images[j], cmap='gray_r')
    if j == 0:
        ax.set_ylabel("Original", rotation=0, labelpad=40, va='center')
    ax.set_xticks([])
    ax.set_yticks([])

# Rows for reconstructions
for row_idx, n in enumerate(n_values, start=1):
    for col_idx in range(num_samples):
        ax = plt.subplot(len(n_values) + 1, num_samples,
                         row_idx * num_samples + col_idx + 1)
        recon = reconstruct_digit(col_idx, scores, pca, n)
        ax.imshow(recon, cmap='gray_r')
        if col_idx == 0:
            ax.set_ylabel(f"n={n}", rotation=0, labelpad=40, va='center')
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle("Digit Reconstructions with Different Numbers of PCA Components")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("outputs/pca_reconstructions.png")
plt.close()
print(f"PCA Reconstructions plot saved to assignments_03/outputs")

# The digits become clearly recognizable somewhere around n=15–40 components.
# This roughly matches where the cumulative variance curve starts to level off,
# meaning most of the important structure is captured by those components.
