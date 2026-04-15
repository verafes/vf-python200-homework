# Week 2 mini-project: predict student math G3 from background features.
import os

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # assignments_02
DATA_DIR = os.path.join(BASE_DIR, "..", "assignments", "resources")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# data from Feature Guide
FEATURES_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "absences", "freetime", "goout"
]
BINARY_COLS = ["schoolsup", "internet", "higher", "activities", "sex"]


# Task 1 - Load & Explore Data
print("\n--- Task 1: Load & Explore Data ---")

CSV_PATH = os.path.join(DATA_DIR, "student_performance_math.csv")
if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, sep=";")

print(f"Loaded dataset: {df.shape}")
print(f"\nFirst 5 rows: {df.head()}")
print(f"\nData types: {df.dtypes}")

# Check missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"\nMissing values per column:\n {missing[missing > 0]}")

# Histogram of G3
plt.figure(figsize=(8, 5))
plt.hist(df["G3"], bins=21, color="orange", edgecolor="lightgray")
plt.title("Distribution of Final Math Grades")
plt.xlabel("G3 (Final Grade)")
plt.ylabel("Count")
plt.savefig(f"{OUTPUT_DIR}/g3_distribution.png")
plt.close()
print(f"\nG3 histogram saved to {OUTPUT_DIR}")

# The histogram shows a big spike at G3 = 0. These zeros are not real grades,
# they represent students who didn’t take the final exam. That’s why they sit apart
# from the rest of the grade distribution (grades 1–20)--it visually confirms
# that they are missing targets, not valid low grades.


# --- Task 2: Preprocess Data ---
print("\n--- Task 2: Preprocess Data ---")
print(f"Original shape: {df.shape}")

# Verify G3 exists
if "G3" not in df.columns:
    raise ValueError("Column 'G3' missing — dataset is invalid.")

# Checking how many students have G3 = 0 (absent from exam)
g3_zero_count = (df["G3"] == 0).sum()
print(f"\nNumber of students with G3=0 (missed exam): {g3_zero_count}" )

df_original = df.copy()

# Remove G3=0
df_clean = df[df["G3"] != 0].copy()
print(f"\nFiltered shape: {df_clean.shape}")

# Check unique values before conversion
print(f"\nUnique values in 'sex': {df['sex'].astype(str).unique()}")
print(f"\nUnique values in 'schoolsup': {df['schoolsup'].astype(str).unique()}")

# Why remove G3 = 0 rows?
# As students with G3=0 did not take the final exam, their "grade" is not actual score.
# Keeping them would confuse the model: it would treat "no final grade" as if the
# student actually scored zero, which would drag the data downward and would distort
# the relationship between features and real final grades.

# Convert yes/no to 1/0 and F/M to 0/1
yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
df_original[yes_no_cols] = (df_original[yes_no_cols].replace({"yes": 1, "no": 0}).astype(int))
df_clean[yes_no_cols] = (df_clean[yes_no_cols].replace({"yes": 1, "no": 0}).astype(int))

df_original["sex"] = (df_original["sex"].replace({"F": 0, "M": 1}).astype(int))
df_clean["sex"] = (df_clean["sex"].replace({"F": 0, "M": 1}).astype(int))

# Correlation check: absences vs G3
corr_original = df_original["absences"].corr(df_original["G3"])
corr_filtered = df_clean["absences"].corr(df_clean["G3"])

print(f"\nCorr(absences, G3) original dataset:  {corr_original:.4f}")
print(f"Corr(absences, G3) filtered dataset: {corr_filtered:.4f}")

# Why does the absences–G3 correlation change?
# In the original data, many students with G3 = 0 also have very high absences.
# Because all of them share the same final grade (0), their absences don't line up
# with different G3 values, so the correlation looks weaker.
# After we remove these "no-show" students, absences line up more clearly
# with changes in G3, so the correlation becomes stronger.


# --- Task 3: EDA ---
print("\n--- Task 3: Exploratory Data Analysis ---")

numeric_cols = (
    df_clean.select_dtypes(include=["number"])
    .drop(columns=["G1", "G2", "G3"], errors="ignore")
    .columns.tolist()
)
print(f"Numeric_cols: {numeric_cols}")

present_cols = set(FEATURES_COLS + BINARY_COLS).issubset(numeric_cols)
print("All required columns present:", present_cols)

# correlations with G3
correlations = {}
for col in numeric_cols:
    r, p = pearsonr(df_clean[col], df_clean["G3"])
    correlations[col] = r

sorted_corr = sorted(correlations.items(), key=lambda x: x[1])

print("\n2 Correlations with G3 (sorted):")
for col, val in sorted_corr:
    num = f"{val:.6f}"
    print(f"{col.ljust(10)} {num.rjust(10)}")

# The strongest relationship is usually "failures" (negative correlation),
# meaning more past failures → lower final grade.
# Scatter plots are chosen to show relationships between two numeric variables.
# absences vs G3 and failures vs G3


# Visualization 1: Study time vs G3
plt.figure(figsize=(6,4))
plt.scatter(df_clean["studytime"], df_clean["G3"], alpha=0.5)
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.savefig(f"{OUTPUT_DIR}/g3_studytime.png")
plt.close()
print(f"\nG3 vs Study time scatter saved to {OUTPUT_DIR}")

# Students who report higher weekly study time tend to have slightly higher G3.
# The relationship is positive but not very strong, which matches the correlation value.
# though the relationship is not perfectly linear.

# Visualization 2: Internet Access vs G3
plt.style.use("default")
plt.figure()
ax = df_clean.boxplot(column="G3", by="internet", color="green", grid=True)
ax.yaxis.grid(True, color="#EAEAEA", linewidth=0.5)
ax.xaxis.grid(True, color="#EAEAEA", linewidth=0.5)
plt.title("Final Grade (G3) by Internet Access")
plt.suptitle("")
plt.xticks([1, 2], ["0 = No Internet", "1 = Yes Internet"])
plt.ylabel("Final Grade (G3)")
plt.xlabel("Internet access")
plt.savefig(f"{OUTPUT_DIR}/g3_internet.png")
plt.close()
print(f"\nG3 vs Internet access boxplot saved to {OUTPUT_DIR}")

# Students with internet access tend to have slightly higher G3 scores.
# The difference is small, matching the weak positive correlation.


# --- Task 4: Baseline Model ---
print("\n--- Task 4: Baseline Model ---")

# Feature and target
X = df_clean[["failures"]]
y = df_clean["G3"]

# Train/test split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression().fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
slope = model.coef_[0]
intercept = model.intercept_
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# The slope shows how much G3 changes per additional failure.
# Since grades are 0–20, a negative slope means each failure reduces expected score.
# RMSE shows average prediction error in grade points.
# R2 shows how much variance is explained — usually low here because one feature is not enough.


# --- Task 5: Build the Full Model ---
print("\n--- Task 5: Full Model ---")
feature_cols = [
    "failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
    "internet", "sex", "freetime", "activities", "traveltime"
]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_train = model.score(X_train, y_train)
r2_test = model.score(X_test, y_test)

print(f"Train R2: {r2_train:.4f}")
print(f"Test R2: {r2_test:.4f}")
print(f"RMSE: {rmse:.4f} \n")

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s} {coef:+.4f}")


# --- Task 6: Predicted vs Actual Plot ---
print("\n--- Task 6: Predicted vs Actual ---")

plt.figure()
plt.scatter(y_pred, y_test, color="gold", alpha=0.6)

# diagonal reference line
lo = min(y_test.min(), y_pred.min())
hi = max(y_test.max(), y_pred.max())
plt.plot([lo, hi], [lo, hi], color="seagreen", linestyle="--", linewidth=2)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/predicted_vs_actual.png")
print(f"Predicted vs Actual scatter saved to {OUTPUT_DIR}")


# Adding G1 makes the model much stronger. The test R2 jumps from about 0.15 to around 0.80.
# This does NOT mean that G1 causes G3. It just means that students who do well
# early in the year usually do well at the end too.
# G1 and G3 are basically measuring the same thing at two different times in the year.

# Because of that, this model is not very helpful for finding students who might struggle early.
# By the time G1 is available, the student has already finished a full grading period.
# So the model is only telling us what the teacher already knows from the first grade.

# If teachers want to help students before they fall behind, they need information
# that comes earlier than G1 — things like attendance patterns, study habits,
# support at home, or past failures. Those early signals are more matter useful
# for early intervention than a grade that already reflects how student is doing.
