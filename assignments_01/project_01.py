import os

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

from prefect import flow, task, get_run_logger

# Folder paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "assignments", "resources", "happiness_project")
OUTPUT_DIR = os.path.join(BASE_DIR, "assignments_01", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# global vars
HAPPINESS_COL = "happiness_score"
REGION = "regional_indicator"

# Helper
def normalize_columns(df):
    # standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # --- normalize happiness_score ---
    if HAPPINESS_COL not in df.columns:
        if "ladder_score" in df.columns:
            df.rename(columns={"ladder_score": HAPPINESS_COL}, inplace=True)
        elif "score" in df.columns:
            df.rename(columns={"score": HAPPINESS_COL}, inplace=True)

    # --- normalize gdp_per_capita ---
    gdp_col = next((col for col in df.columns if "gdp" in col), None)
    if gdp_col:
        df["gdp_per_capita"] = df[gdp_col]

    return df

# Task 1 - Load & Merge Data
@task(retries=3, retry_delay_seconds=2)
def load_and_merge_data():
    """
    Load all yearly World Happiness CSV files, normalize column names,
    add a 'year' column, merge them, and save the combined dataset.
    """
    logger = get_run_logger()
    logger.info('Reading CSV files')

    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        logger.error(f"Directory missing: {data_dir.absolute()}")
        return

    all_dfs = []

    for year in range(2015, 2025):
        file_path = f"{DATA_DIR}/world_happiness_{year}.csv"

        try:
            logger.info(f"Loading file: {file_path}")
            df = pd.read_csv(file_path, sep=';', decimal=',')
        except FileNotFoundError:
            logger.error(f"Missing file for {year}")
            continue
        except Exception:
            logger.warning(f"Parsing issue in {file_path}; using fallback parser.")
            df = pd.read_csv(file_path, sep=";", decimal=",", engine="python", on_bad_lines="skip")

        df = normalize_columns(df)
        df["year"] = year

        all_dfs.append(df)
        logger.info(f"Loaded {year}: {len(df)} rows, columns: {df.columns.tolist()}")

    merged_df = pd.concat(all_dfs, ignore_index=True)

    output_path = f"{OUTPUT_DIR}/merged_happiness.csv"
    merged_df.to_csv(output_path, index=False)

    logger.info(f"Merged data saved to {output_path}")
    logger.info(f"Saved merged dataset to {OUTPUT_DIR}  ({len(merged_df)} rows total)")
    logger.info(f"Final columns: {merged_df.columns.tolist()}")

    return merged_df


# Task 2 - Descriptive Statistics
@task
def compute_statistics(df):
    """
    Compute and log basic statistics for happiness_score,
    plus averages by year and region.
    """
    logger = get_run_logger()

    if HAPPINESS_COL not in df.columns:
        raise KeyError(f"Column '{HAPPINESS_COL}' not found in DataFrame.")

    score = df[HAPPINESS_COL]
    # Overall stats
    mean_score = score.mean()
    median_score = score.median()
    std_score = score.std()

    logger.info(f"Mean happiness score: {mean_score:.4f}")
    logger.info(f"Median happiness score: {median_score:.4f}")
    logger.info(f"Std dev happiness score: {std_score:.4f}")

    # Group by year
    yearly_avg = df.groupby("year")[HAPPINESS_COL].mean().sort_index()
    formatted_yearly_avg = yearly_avg.apply(lambda x: f"{x:.4f}")
    logger.info(f"\n--- Average happiness score by YEAR:\n{formatted_yearly_avg} ")

    # Group by region
    if REGION in df.columns:
        regional_avg = df.groupby(REGION)[HAPPINESS_COL].mean().sort_values(ascending=False)
        formatted_regional_avg = regional_avg.apply(lambda x: f"{x:.4f}")
        logger.info(f"\n--- Average happiness by REGION:\n{formatted_regional_avg}")
    else:
        logger.warning(f"Column {REGION} not found; skipping regional statistics.")


# Task 3 - Visual Exploration
@task
def create_visualizations(df):
    """
    Generate histogram, boxplot, scatter plot, and correlation heatmap,
    saving all figures to the outputs directory.
    """
    logger = get_run_logger()

    # Histogram of happiness scores
    plt.figure(figsize=(8, 6))
    sns.histplot(df[HAPPINESS_COL].dropna(), bins=30, kde=False, color="skyblue", linewidth=0)
    plt.title("Happiness Score Distribution (All Years)")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.savefig(f"{OUTPUT_DIR}/happiness_histogram.png")
    plt.close()
    logger.info("Saved happiness_histogram.png")

    # Boxplot by year
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="year", y=HAPPINESS_COL, data=df, color="lightgreen")
    plt.title("Happiness Scores by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/happiness_by_year.png")
    plt.close()
    logger.info("Saved happiness_by_year.png")

    # Scatter GDP vs Happiness
    plt.figure()
    plt.scatter(df["gdp_per_capita"], df[HAPPINESS_COL], color="orange")
    plt.title("GDP vs Happiness")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    plt.savefig(f"{OUTPUT_DIR}/gdp_vs_happiness.png")
    plt.close()
    logger.info("Saved gdp_vs_happiness.png")

    # Correlation heatmap
    numeric_df = df.select_dtypes(include="number").drop(
        columns=["ranking", "year"],
        errors="ignore"
    )
    corr = numeric_df.select_dtypes(include="number").corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.close()
    logger.info("Saved correlation_heatmap.png")


# Task 4 - Hypothesis testing
@task
def hypothesis_tests(df):
    """
    Run t-tests comparing 2019 vs 2020 and two regions,
    returning test statistics and interpretations.
    """
    logger = get_run_logger()
    if HAPPINESS_COL not in df.columns or "year" not in df.columns:
        raise KeyError(f"Required columns '{HAPPINESS_COL}' and/or 'year' not found.")

    data_2019 = df.loc[df["year"] == 2019, HAPPINESS_COL].dropna()
    data_2020 = df.loc[df["year"] == 2020, HAPPINESS_COL].dropna()
    mean_2019 = data_2019.mean()
    mean_2020 = data_2020.mean()

    t_stat, p_value = stats.ttest_ind(data_2019, data_2020)

    logger.info("--- T-test: happiness_score 2019 vs 2020: ---")
    logger.info(f"   2019 mean: {mean_2019:.4f}")
    logger.info(f"   2020 mean: {mean_2020:.4f}")
    logger.info(f"   T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

    if p_value < 0.05:
        direction = "lower" if mean_2020 < mean_2019 else "higher"
        ttest_summary = ("   At alpha < 0.05, there is evidence that average happiness changed, "
                  f"with scores being significantly {direction} in 2020 compared to 2019. "
                  )
    else:
        ttest_summary = "   At alpha >= 0.05, no strong evidence that happiness scores changed from 2019 to 2020."
    logger.info(ttest_summary)

    # Second test (North America and ANZ vs East Asia)
    region_A = "North America and ANZ"
    region_B = "East Asia"
    data_A = df.loc[df[REGION] == region_A][HAPPINESS_COL].dropna()
    data_B = df.loc[df[REGION] == region_B][HAPPINESS_COL].dropna()

    t_stat2, p_value2 = stats.ttest_ind(data_A, data_B, equal_var=False)
    mean_A = data_A.mean()
    mean_B = data_B.mean()

    logger.info(f"{region_A} mean: {mean_A:.4f}")
    logger.info(f"{region_B} mean: {mean_B:.4f}")
    logger.info(f"   T-statistic: {t_stat2:.4f}, P-value: {p_value2:.4f}")

    if p_value2 < 0.05:
        direction2 = "higher" if mean_A > mean_B else "lower"
        result2 = (
            f"   At α = 0.05, {region_A} has significantly {direction2} happiness "
            f"scores than {region_B}. This difference is unlikely to be due to chance."
        )
    else:
        result2 = (
            f"   No strong evidence of a difference in happiness between {region_A} and {region_B}."
        )
    logger.info(result2)

    return {
        "years_2019_2020": (t_stat, p_value),
        "mean_2019": mean_2019,
        "mean_2020": mean_2020,
        "regions": (t_stat2, p_value2)}


# Task 5: Correlation and Multiple Comparisons (Bonferroni)
@task
def correlation_analysis(df) -> dict:
    """
    Compute Pearson correlations with happiness_score,
    apply Bonferroni correction, and report significant variables.
    """
    logger = get_run_logger()

    numeric_df = df.select_dtypes(include="number").copy()
    cols_to_drop = ["ranking", "year", HAPPINESS_COL]
    existing = [c for c in cols_to_drop if c in numeric_df.columns]
    numeric_df = numeric_df.drop(columns=existing)
    numeric_cols = list(numeric_df.columns)

    logger.info("--- Pearson correlations with happiness_score ---")
    correlations = {}
    p_values = {}
    for col in numeric_cols:
        x = df[col].dropna()
        y = df.loc[x.index, HAPPINESS_COL].dropna()

        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]

        if len(x) < 3:
            logger.warning(f"Not enough data to compute correlation for {col}; skipping.")
            continue

        r, p = stats.pearsonr(x, y)
        correlations[col] = r
        p_values[col] = p
        logger.info(f"  {col}: r={r:.4f}, p={p:.4f}")

    num_tests = len(correlations)
    if num_tests == 0:
        logger.warning("No correlation tests performed.")
        return {
            "correlations": correlations,
            "p_values": p_values,
            "adjusted_alpha": None,
            "significant_original": [],
            "significant_bonferroni": [],
            "strongest_var": None,
        }

    alpha = 0.05
    adjusted_alpha = alpha / num_tests

    logger.info(f"Performed {num_tests} correlation tests.")
    logger.info(f"Original alpha: {alpha}, Bonferroni-adjusted alpha: {adjusted_alpha:.5f}")

    significant_original = [col for col, p in p_values.items() if p < alpha]
    significant_bonferroni = [col for col, p in p_values.items() if p < adjusted_alpha]

    logger.info("--- Correlation with happiness_score: ----")

    strongest_var = None
    if significant_bonferroni:
        strongest_var = max(significant_bonferroni, key=lambda c: abs(correlations[c]))
        strongest_r = correlations[strongest_var]
        logger.info(
            f"Most strongly correlated variable after Bonferroni: {strongest_var} (r = {strongest_r:.4f})"
        )
    else:
        logger.info("No variables remain significant after Bonferroni correction.")

    return {
        "correlations": correlations,
        "p_values": p_values,
        "adjusted_alpha": adjusted_alpha,
        "significant_original": significant_original,
        "significant_bonferroni": significant_bonferroni,
        "strongest_var": strongest_var,
    }


# Task 6: Summary Report
@task
def summary_report(df, hypo_results, corr_data):
    """
    Log a human-readable summary:
    dataset size, top/bottom regions, t-test result, and strongest corrected correlation.
    """
    logger = get_run_logger()
    logger.info("------------------------------------------------")
    logger.info("--- Summary report: World Happiness Analysis ---")
    logger.info("------------------------------------------------")

    unique_years = df["year"].nunique() if "year" in df.columns else None
    unique_countries = df["country"].nunique() if "country" in df.columns else None

    logger.info(f"Total number of countries in dataset: {unique_countries}")
    logger.info(f"Total number of years in dataset: {unique_years}")

    # Top 3 and bottom 3 regions by mean happiness
    region_col = "region" if "region" in df.columns else REGION
    if region_col in df.columns:
        mean_by_region = df.groupby(region_col)[HAPPINESS_COL].mean().sort_values(ascending=False)
        top3 = mean_by_region.head(3)
        bottom3 = mean_by_region.tail(3)

        logger.info("Top 3 regions by mean happiness_score:")
        for region, value in top3.items():
            logger.info(f"  {region}: {value:.4f}")

        logger.info("Bottom 3 regions by mean happiness_score:")
        for region, value in bottom3.items():
            logger.info(f"  {region}: {value:.4f}")
    else:
        logger.info("The 'region' column not found; cannot compute top/bottom regions.")

    # Result of 2019 vs 2020 t-test in plain language
    t_stat, p_val = hypo_results.get("years_2019_2020")
    mean_2019 = hypo_results["mean_2019"]
    mean_2020 = hypo_results["mean_2020"]
    alpha = 0.05

    logger.info(f"2019 vs 2020 happiness comparison (t-test):")
    if p_val < alpha:
        ttest_summary = (
            f"Between 2019 and 2020, mean happiness changed from {mean_2019:.3f} to {mean_2020:.3f}. "
            f"Pandemic Impact: Statistically significant, p = {p_val:.4f}, t-stat: {t_stat:.4f}. "
            "Data confirms a measurable change in global well-being during the first year of the pandemic."
        )
    else:
        ttest_summary = (
            f"Between 2019 and 2020, mean happiness changed from {mean_2019:.3f} to {mean_2020:.3f}. "
            f"Pandemic Impact: NOT significant, p = {p_val:.4f}, t-stat: {t_stat:.4f}. "
            "There is no clear evidence of a real change; the observed small difference could be random."
        )
    logger.info(ttest_summary)

    # Variable most strongly correlated with happiness after Bonferroni
    strongest_var = corr_data.get("strongest_var")
    strongest_r = corr_data["correlations"][strongest_var]

    if strongest_var is not None:
        logger.info(
            f"After Bonferroni correction, the metric most strongly correlated with "
            f"happiness_score is {strongest_var} with r = {strongest_r:.4f}."
        )
    else:
        logger.info(
            "No explanatory variables remain significantly correlated with happiness_score "
            "after Bonferroni correction."
        )


@flow
def happiness_pipeline():
    """
    Run the full analysis: load data, compute stats, visualize, test, and report.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_merge_data()
    compute_statistics(df)
    create_visualizations(df)
    hyp_results = hypothesis_tests(df)
    corr_results = correlation_analysis(df)
    summary_report(df, hyp_results, corr_results)


if __name__ == "__main__":
    happiness_pipeline()
