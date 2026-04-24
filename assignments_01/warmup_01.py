# Warmup Exercises — Part 1

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns


os.makedirs("outputs", exist_ok=True)

def tasks_part_one():
    # --- Pandas ---

    print("\n--- Pandas Q1 ---")
    # Create DataFrame and print first 3 rows, shape, and dtypes of each column
    data = {
        "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
        "grade":  [85, 72, 90, 68, 95],
        "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
        "passed": [True, True, True, False, True]
    }

    df = pd.DataFrame(data)
    print(df.head(3))
    print(f"\nShape: {df.shape} \n")

    print(f"Data types:\n {df.dtypes}")

    print("\n--- Pandas Q2 ---")
    # Filter: passed == True AND grade > 80
    filtered = df[(df["passed"] == True) & (df["grade"] > 80)]
    print("Students who passed and have grade > 80:")
    print(filtered)

    print("\n--- Pandas Q3 ---")
    # Add curved grade column (+5 points)
    df["grade_curved"] = df["grade"] + 5
    print("DataFrame with curved grades:")
    print(df)

    print("\n--- Pandas Q4 ---")
    # Add uppercase name column using .str accessor
    df["name_upper"] = df["name"].str.upper()
    print("Name and Name Upper:")
    print(df[["name", "name_upper"]])

    print("\n--- Pandas Q5 ---")
    # Group by city and compute mean grade
    mean_by_city = df.groupby("city")["grade"].mean()
    print("Mean grade by city:")
    print(mean_by_city)

    print("\n--- Pandas Q6 ---")
    # Replace "Austin" with "Houston"
    df["city"] = df["city"].replace("Austin", "Houston")
    print("Updated cities:")
    print(df[["name", "city"]])

    print("\n--- Pandas Q7 ---")
    # Sort by grade descending, print top 3
    sorted_by_grade = df.sort_values(by="grade", ascending=False)
    print("Top 3 students by grade:")
    print(sorted_by_grade.head(3))

    # --- NumPy ---

    print("\n--- NumPy Q1 ---")
    # Create 1D array and print shape, dtype, ndim
    arr1D = np.array([10, 20, 30, 40, 50])
    print(f"Array 1D: {arr1D}")
    print(f"Shape: {arr1D.shape}")
    print(f"Dtype: {arr1D.dtype}")
    print(f"ndim: {arr1D.ndim}")

    print("\n--- NumPy Q2 ---")
    # Create 2D array and print shape + size (total elements)
    arr2D = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    print("Array 2D:\n", arr2D)
    print(f"Shape: {arr2D.shape}")
    print(f"Size: {arr2D.size}")

    print("\n--- NumPy Q3 ---")
    # Slice top-left 2x2 block
    slice_2x2 = arr2D[:2, :2]
    print("Top-left 2x2 block:")
    print(slice_2x2)

    print("\n--- NumPy Q4 ---")
    # Create zeros (3x4) and ones (2x5)
    zeros_arr = np.zeros((3, 4))
    ones_arr = np.ones((2, 5))
    print("Zeros array:\n", zeros_arr)
    print("Ones array:\n", ones_arr)

    print("\n--- NumPy Q5 ---")
    # np.arange(0, 50, 5) + stats
    arr_range = np.arange(0, 50, 5)
    print(f"Array: {arr_range}")
    print(f"Shape: {arr_range.shape}")
    print(f"Mean: {arr_range.mean()}")
    print(f"Sum: {arr_range.sum()}")
    print(f"Std Dev: {arr_range.std()}")

    print("\n--- NumPy Q6 ---")
    # Generate 200 random normal values (mean=0, std=1)
    random_values = np.random.normal(loc=0, scale=1, size=200)
    print(f"Mean: {random_values.mean()}")
    print(f"Std Dev: {random_values.std()}")

    # --- Matplotlib ---

    #--- Matplotlib Q1 ---
    # Line plot of squares

    x = [0, 1, 2, 3, 4, 5]
    y = [0, 1, 4, 9, 16, 25]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title("Squares")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.savefig("outputs/matplotlib_q1_squares_plot.png")
    plt.show()

    #--- Matplotlib Q2 ---
    # Line plot of squares
    subjects = ["Math", "Science", "English", "History"]
    scores   = [88, 92, 75, 83]

    plt.figure()
    plt.bar(subjects, scores, color="skyblue")
    plt.title("Subject Scores")
    plt.xlabel("Subjects")
    plt.ylabel("Scores")
    plt.savefig("outputs/matplotlib_q2_subject_scores_plot.png")
    plt.show()

    #--- Matplotlib Q3 ---
    # Scatter plot of two datasets

    x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
    x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

    plt.figure()
    plt.scatter(x1, y1, label="Dataset 1", color="red")
    plt.scatter(x2, y2, label="Dataset 2", color="blue")
    plt.title("Scatter Plot Comparison")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("outputs/matplotlib_q3_scatter_plot_comparison.png")
    plt.show()

    #--- Matplotlib Q4 ---
    # Subplots: line plot + bar plot

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Left line subplot
    axes[0].plot(x, y, marker="o")
    axes[0].set_title("Squares Line Plot")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # Right bar subplot
    axes[1].bar(subjects, scores, color="orange")
    axes[1].set_title("Subject Scores")
    axes[1].set_xlabel("Subject")
    axes[1].set_ylabel("Score")

    plt.tight_layout()
    plt.savefig("outputs/matplotlib_q4_subplots.png")
    plt.show()

    # --- Descriptive Statistics ---
    # Descriptive Stats Question 1
    print("\n--- Descriptive Stats Q1 ---")
    # Mean, median, variance, std

    data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

    mean_val = np.mean(data)
    median_val = np.median(data)
    variance_val = np.var(data)
    std_val = np.std(data)

    print(f"Mean: {mean_val}")
    print(f"Median: {median_val}")
    print(f"Variance: {variance_val}")
    print(f"Standard Deviation: {std_val}")

    # Descriptive Stats Q2
    # Histogram of normal distribution

    scores = np.random.normal(65, 10, 500)

    plt.figure()
    plt.hist(scores, bins=20, color="purple", edgecolor="black")
    plt.title("Distribution of Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig("outputs/descriptive_stats_q2_histogram.png")
    plt.show()

    # Descriptive Stats Q3
    # Boxplot comparing two groups
    group_a = [55, 60, 63, 70, 68, 62, 58, 65]
    group_b = [75, 80, 78, 90, 85, 79, 82, 88]

    plt.figure()
    plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])

    plt.title("Score Comparison")
    plt.ylabel("Scores")
    plt.savefig("outputs/descriptive_stats_q3_boxplot_groups.png")
    plt.show()

    # Descriptive Stats Q4
    # Boxplots: normal vs exponential

    normal_data = np.random.normal(50, 5, 200)
    skewed_data = np.random.exponential(10, 200)

    plt.figure()
    plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])
    plt.title("Distribution Comparison")
    plt.ylabel("Values")
    plt.savefig("outputs/descriptive_stats_q4_boxplot.png")
    plt.show()

    """  
    The exponential distribution is more skewed (right-skewed).
    For skewed distributions, the median is a better measure of central tendency.
    For symmetric distributions like the normal distribution, the mean is appropriate.
    """

    # Descriptive Stats Q5
    # Mean, median, mode + explanation
    print("\n--- Descriptive Stats Q5 ---")

    data1 = [10, 12, 12, 16, 18]
    data2 = [10, 12, 12, 16, 150]

    print("Data1 Mean:", np.mean(data1))
    print("Data1 Median:", np.median(data1))
    print("Data1 Mode:", mode(data1))

    print("\nData2 Mean:", np.mean(data2))
    print("Data2 Median:", np.median(data2))
    print("Data2 Mode:", mode(data2))

    """
    The mean of data2 is much larger than the median because the value 150 is an outlier, e.g. value that is far away from the rest.
    Outliers pull the mean much higher and can make data look misleading. But they can indicate errors, rare event, special case.  
    The median is resistant to extreme values and stays almost the same.
    Mode is most frequent value.
    """

    # --- Hypothesis Testing ---

    print("\n--- Hypothesis Q1 ---")
    # Independent samples t-test

    group_a = [72, 68, 75, 70, 69, 73, 71, 74]
    group_b = [80, 85, 78, 83, 82, 86, 79, 84]

    t_stat, p_value = stats.ttest_ind(group_a, group_b)

    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")

    print("\n--- Hypothesis Q2 ---")
    # Significance check at alpha = 0.05
    alpha = 0.05
    if p_value < alpha:
        flag = ""
    else:
        flag= "not "

    print(f"The result is statistically {flag}significant at alpha = 0.05.")

    print("\n--- Hypothesis Q3 ---")
    # Paired t-test
    before = [60, 65, 70, 58, 62, 67, 63, 66]
    after  = [68, 70, 76, 65, 69, 72, 70, 71]

    t_stat_paired, p_val_paired = stats.ttest_rel(before, after)
    print("t-statistic:", t_stat_paired)
    print("p-value:", p_val_paired)


    print("\n--- Hypothesis Q4: One-sample t-test ---")
    # One-sample t-test vs benchmark = 70
    scores = [72, 68, 75, 70, 69, 74, 71, 73]

    t_stat_one, p_val_one = stats.ttest_1samp(scores, 70)
    print("t-statistic:", t_stat_one)
    print("p-value:", p_val_one)


    print("\n--- Hypothesis Q5 ---")
    # One-tailed test: group_a < group_b

    t_stat_one_tailed, p_val_one_tailed = stats.ttest_ind(
        group_a, group_b, alternative='less'
    )
    print("p-value:", p_val_one_tailed)


    print("\n--- Hypothesis Q6 ---")
    # Plain-language conclusion

    print("Group B scored significantly higher than Group A on average; \n"
          "given the extremely low p-value (probability), \n"
          "this difference is very unlikely to have occurred by random chance.")

    # --- Correlation ---
    print("\n--- Correlation Q1 ---")
    # Correlation Matrix
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    corr_matrix = np.corrcoef(x, y)
    print(f"Full Matrix:\n {corr_matrix}")
    print(f"Correlation coefficient (x,y): {corr_matrix[0, 1]}")

    """
    Expected correlation:
    Because y is exactly 2*x, the relationship is perfectly linear and positive.
    Therefore, the correlation should be 1.0.
    """

    print("\n--- Correlation Q2 ---")
    # Pearson Correlation (pearsonr)
    x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
    y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

    corr, p_value = pearsonr(x, y)

    print(f"Correlation coefficient: {corr}")
    print(f"p-value: {p_value}")

    print("\n--- Correlation Q3 ---")
    # DataFrame Correlation Matrix
    people = {
        "height": [160, 165, 170, 175, 180],
        "weight": [55,  60,  65,  72,  80],
        "age":    [25,  30,  22,  35,  28]
    }

    df = pd.DataFrame(people)
    print(f"DataFrame Correlation Matrix:\n {df.corr()}")

    # Correlation Question 4 - Scatter Plot
    x = [10, 20, 30, 40, 50]
    y = [90, 75, 60, 45, 30]

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y)
    plt.title("Negative Correlation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("outputs/correlation_q4_scatter.png")
    plt.show()

    # Correlation Question 5: Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("outputs/correlation_q5_heatmap.png")
    plt.show()

# --- Pipeline  ---

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    """Convert NumPy array to a pandas Series."""
    return pd.Series(arr, name="values")

def clean_data(series):
    """Remove NaN values."""
    return series.dropna()

def summarize_data(series):
    """Return summary statistics as a dictionary."""
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

def data_pipeline(arr):
    """Run the full pipeline: create → clean → summarize."""
    s = create_series(arr)
    cleaned = clean_data(s)

    return summarize_data(cleaned)

# Run pipeline and print results
if __name__ == "__main__":
    tasks_part_one()

    summary = data_pipeline(arr)
    print(f"\nQ1 Plain Python pipeline result:")
    print("\n".join(f"{key}: {value}" for key, value in summary.items()))

