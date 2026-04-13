# --- scikit-learn API ---
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)


# --- scikit-learn API ---

print(f"--- scikit-learn Q1 ---")
# Q1: Reshaping a 1D array into 2D for scikit-learn

# Data
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)

# Predictions
pred_4y = model.predict(np.array([[4]]))[0]
pred_8y = model.predict(np.array([[8]]))[0]

print(f"Slope (coef): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Predicted salary for 4 years: {pred_4y:.2f}")
print(f"Predicted salary for 8 years: {pred_8y:.2f}")


print(f"\n--- scikit-learn Q2 ---")
# Q2: Reshaping a 1D array into 2D for scikit-learn
x = np.array([10, 20, 30, 40, 50])
print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)
print("Reshaped to 2D:", x_2d.shape)

# scikit-learn expects X to be 2D because it reads data as a table
# and represents a matrix of samples and features.
# Even if we have only one feature, scikit-learn still needs a column structure:
# rows = samples, columns = features


# scikit-learn Q3
print(f"\n--- scikit-learn Q3 ---")
# Q3: K-Means clustering (create → fit → predict)
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

# Create the model
kmeans = KMeans(n_clusters=3, random_state=42)
# Fit the model
kmeans.fit(X_clusters)
# Predict cluster labels
labels = kmeans.predict(X_clusters)

print("Cluster centers:\n", kmeans.cluster_centers_)
print("Points per cluster:", np.bincount(labels))

plt.figure(figsize=(8, 6))

# --- Plot ---
# Scatter plot colored by cluster label
plt.scatter(
    X_clusters[:, 0],
    X_clusters[:, 1],
    c=labels,
    cmap="plasma",
    s=40,
    alpha=0.8
)

# Plot cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    marker="x",
    s=100
)

plt.title("K-Means Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Save figure
plt.savefig("outputs/kmeans_clusters.png")
plt.show()
plt.close()

# --- Linear Regression ---

# Dataset
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Q1 Linear Regression: Scatter Plot

plt.figure(figsize=(8, 6))
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Cost")
plt.colorbar(label="Smoker (1 = smoker)")
plt.tight_layout()

plt.savefig("outputs/cost_vs_age.png")
plt.show()

# The plot shows two clear groups: Smokers (red points) have consistently higher costs,
# while Non‑smokers (blue points) have lower cost at every age.
# This indicates smoker strongly increases medical cost beyond what age alone can explain.

print(f"\n--- Linear Regression Q2---")
# Q2 Train/Test Split (age only)

X = age.reshape(-1, 1)  # age as 2D feature
y = cost

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# outcomes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


print(f"\n--- Linear Regression Q3---")
# Q3: Fit model using age only

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# The slope shows how much medical cost increases per additional year of age.
# For example, if slope near 200, each extra year of age adds about $200
# to expected medical annual cost, on average.


print(f"\n--- Linear Regression Q4---")
# Q4: Add smoker feature

X_full = np.column_stack([age, smoker])
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)
r2_full = model_full.score(X_test_f, y_test_f)

print(f"R2 (age only): {r2:.4f}")
print(f"R2 (age + smoker): {r2_full:.4f}")

print(f"age coefficient: {model_full.coef_[0]:.4f}")
print(f"smoker coefficient: {model_full.coef_[1]:.4f}")

# Adding smoker feature improves the model because it explains cost differences better.
# The smoker coefficient shows how much extra cost is added if someone is a smoker
# (around $15,000).


# Linear Regression Q5
# Predicted vs Actual Plot
y_pred_full = model_full.predict(X_test_f)

plt.figure()
plt.scatter(y_pred_full, y_test_f)

# Diagonal line
min_val = min(y_test_f.min(), y_pred_full.min())
max_val = max(y_test_f.max(), y_pred_full.max())
plt.plot([min_val, max_val], [min_val, max_val], color="orange")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.savefig("outputs/predicted_vs_actual.png")
plt.show()

# Points above the diagonal `mean actual > predicted` (model underestimates).
# Points below `mean predicted > actual` (model overestimates)
