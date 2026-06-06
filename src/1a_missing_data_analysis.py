"""
Missing Data Analysis & Imputation
====================================
Analyzes missingness patterns for all features with > 5% missing values,
determines whether missingness is MCAR or MAR via correlation analysis,
and applies the appropriate imputation strategy per feature.

Imputation strategy:
  - Numeric features:   KNN imputation (handles both MCAR and MAR)
  - Categorical features: Most-frequent (mode) imputation

Run this as Step 5.5 inside clean(), after computing all derived features
and before renaming columns.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer


# ── 1. Identify features with > 5% missing ───────────────────────────────────
def get_high_missing_cols(df: pd.DataFrame, threshold: float = 0.0) -> dict:
    """
    Returns a dict of {col: missing_pct} for all columns exceeding threshold.
    By default threshold=0.0 → returns any column with any missing values.
    Splits into numeric and categorical groups.
    """
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > threshold].sort_values(ascending=False)

    numeric_missing = {}
    categorical_missing = {}

    for col, pct in high_missing.items():
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_missing[col] = round(pct * 100, 1)
        else:
            categorical_missing[col] = round(pct * 100, 1)

    print(f"\n── Features with > {threshold*100:.0f}% missing ──────────────────────────────")
    print(f"\n  Numeric ({len(numeric_missing)}):")
    for col, pct in numeric_missing.items():
        print(f"    {col:<45} {pct:>5}% missing")
    print(f"\n  Categorical ({len(categorical_missing)}):")
    for col, pct in categorical_missing.items():
        print(f"    {col:<45} {pct:>5}% missing")

    return {"numeric": numeric_missing, "categorical": categorical_missing}

    for col in df_corr.columns:
        if df_corr[col].dtype == 'object':
        
            if isinstance(sample, str) and len(str(sample)) > 50:
                print(f'CORRUPTED COLUMN: {col}')
                print(f'  Sample: {str(sample)[:100]}')
    

# ── 2. MAR correlation analysis ───────────────────────────────────────────────
def analyze_missingness(df: pd.DataFrame, missing_cols: dict) -> dict:
    """
    For each column with high missingness, creates a binary missingness
    indicator and correlates it against demographic features.

    Returns a dict of {col: "MAR" | "MCAR"} classification.
    MAR threshold: any absolute correlation > 0.1 with another variable.
    """
    all_high_missing = list(missing_cols["numeric"].keys()) + list(missing_cols["categorical"].keys())

    if not all_high_missing:
        print("\nNo features exceed the missing threshold — skipping analysis.")
        return {}

    # Only correlate missingness against a small set of demographic features
    # Using raw column names since rename hasn't happened yet
    demographic_cols = [
        'RIDAGEYR',   # age
        'RIAGENDR',   # sex
        'RIDRETH3',   # race/ethnicity
        'DMDEDUC2',   # education
        'INDFMPIR',   # poverty-income ratio
        'DMDMARTZ',   # marital status
        'BMXBMI',     # BMI — included as a key health demographic
    ]

    # Keep only demographic cols that exist in the DataFrame and are not high-missing
    reference_cols = [c for c in demographic_cols if c in df.columns and c not in all_high_missing]
    df_ref = df[reference_cols].copy()

    # Encode any categorical reference columns
    for col in df_ref.columns:
        if not pd.api.types.is_numeric_dtype(df_ref[col]):
            df_ref[col] = LabelEncoder().fit_transform(df_ref[col].astype(str))
        else:
            df_ref[col] = pd.to_numeric(df_ref[col], errors='coerce')

    print(f'Demographic reference columns used for MAR analysis: {reference_cols}')
    print(f"\n── MAR Correlation Analysis ─────────────────────────────────────────")
    print(f"  Threshold: |correlation| > 0.1 with any demographic variable → MAR")
    print(f"  Otherwise → MCAR\n")

    mar_results = {}
    correlation_details = {}

    for col in all_high_missing:
        # Binary missingness indicator: 1 = missing, 0 = observed
        missing_indicator = df[col].isna().astype(int)

        if missing_indicator.sum() == 0:
            mar_results[col] = "NONE"
            continue

        # Correlate missingness indicator against demographic reference columns
        corr_series = df_ref.corrwith(missing_indicator).abs().dropna().sort_values(ascending=False)

        max_corr     = corr_series.max()
        top_predictor = corr_series.idxmax() if not corr_series.empty else "none"
        classification = "MAR" if max_corr > 0.1 else "MCAR"

        mar_results[col]       = classification
        correlation_details[col] = corr_series

        # Print top 5 correlated demographic variables
        print(f"  {col}")
        print(f"    → {classification}  (max |r| = {max_corr:.3f} with '{top_predictor}')")
        top5 = corr_series.head(5)
        for predictor, r in top5.items():
            print(f"       {predictor:<40} r = {r:.3f}")
        print()

    # Summary table
    print("── Classification Summary ───────────────────────────────────────────")
    mar_count  = sum(1 for v in mar_results.values() if v == "MAR")
    mcar_count = sum(1 for v in mar_results.values() if v == "MCAR")
    print(f"  MAR  (use KNN / iterative imputation): {mar_count} features")
    print(f"  MCAR (mean/median imputation valid):   {mcar_count} features")

    return mar_results


# ── 3. Impute numeric features ────────────────────────────────────────────────
def impute_numeric(df: pd.DataFrame, numeric_missing_cols: list,
                   n_neighbors: int = 5) -> pd.DataFrame:
    """
    Applies KNN imputation to all numeric columns with high missingness.
    KNN is valid for both MAR and MCAR, so we use it for all numeric columns
    regardless of classification — it's conservative and handles both cases.
    """
    if not numeric_missing_cols:
        print("\n  No numeric columns to impute.")
        return df

    print(f"\n── Numeric Imputation (KNN, k={n_neighbors}) ────────────────────────────────")

    # KNN imputer needs all columns it uses to be numeric
    # Include all numeric columns in the fit so neighbors are computed properly
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Print before stats
    for col in numeric_missing_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            print(f"  {col:<45} missing before: {n_missing:>5} ({n_missing/len(df)*100:.1f}%)")

    # Fit and transform
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_numeric = df[all_numeric_cols].copy()
    df_numeric_imputed = pd.DataFrame(
        imputer.fit_transform(df_numeric),
        columns=all_numeric_cols,
        index=df.index
    )

    # Write imputed values back to df for target columns only
    for col in numeric_missing_cols:
        if col in df_numeric_imputed.columns:
            df[col] = df_numeric_imputed[col]

    # Zero out tiny floating-point noise that is effectively zero
    zeroed = zero_near_zero_values(df)
    if zeroed > 0:
        print(f"  Zeroed {zeroed:,} tiny numeric values to 0")

    # Print after stats
    print()
    for col in numeric_missing_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            print(f"  {col:<45} missing after:  {n_missing:>5}")

    return df


def zero_near_zero_values(df: pd.DataFrame, threshold: float = 1e-12) -> int:
    """
    Replace extremely small numeric values with exact zero.
    This handles values like 5.4e-79 that are effectively numerical noise.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return 0

    tiny_mask = df[numeric_cols].abs() < threshold
    if not tiny_mask.any().any():
        return 0

    zeroed_count = int(tiny_mask.sum().sum())
    df.loc[:, numeric_cols] = df[numeric_cols].where(~tiny_mask, 0)
    return zeroed_count


# ── 4. Impute categorical features ───────────────────────────────────────────
def impute_categorical(df: pd.DataFrame, categorical_missing_cols: list) -> pd.DataFrame:
    """
    Applies most-frequent (mode) imputation to categorical columns.
    Mode imputation is valid for both MCAR and MAR for nominal variables.
    """
    if not categorical_missing_cols:
        print("\n  No categorical columns to impute.")
        return df

    print(f"\n── Categorical Imputation (most frequent / mode) ────────────────────")

    imputer = SimpleImputer(strategy="most_frequent")

    for col in categorical_missing_cols:
        if col not in df.columns:
            continue

        n_missing = df[col].isna().sum()
        mode_val  = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
        print(f"  {col:<45} missing: {n_missing:>5}  → filling with '{mode_val}'")

        # SimpleImputer needs 2D input
        original_dtype = df[col].dtype
        df[[col]] = imputer.fit_transform(df[[col]].astype(str))

        # Restore category dtype if it was one
        if str(original_dtype) == "category":
            df[col] = df[col].astype("category")

    return df


# ── 5. Post-imputation validation ─────────────────────────────────────────────
def validate_imputation(df: pd.DataFrame, original_missing_cols: list):
    """
    Prints a before/after summary of missing values for all imputed columns.
    Warns if any imputed column still has missing values.
    """
    print(f"\n── Post-Imputation Validation ───────────────────────────────────────")
    any_remaining = False

    for col in original_missing_cols:
        if col not in df.columns:
            continue
        remaining = df[col].isna().sum()
        status    = "✓" if remaining == 0 else "⚠ STILL MISSING"
        print(f"  {col:<45} remaining missing: {remaining:>5}  {status}")
        if remaining > 0:
            any_remaining = True

    if any_remaining:
        print("\n  ⚠ Some columns still have missing values after imputation.")
        print("    Consider dropping these rows or using a different strategy.")
    else:
        print("\n  ✓ All targeted columns fully imputed.")


# ── 6. Master function — run everything ───────────────────────────────────────
def analyze_and_impute(df: pd.DataFrame, threshold: float = 0.0,
                       knn_neighbors: int = 5) -> pd.DataFrame:
    """
    Full pipeline:
      1. Identify columns with > threshold% missing
      2. Run MAR correlation analysis
      3. Impute numeric columns with KNN
      4. Impute categorical columns with mode
      5. Validate results

    Returns the imputed DataFrame.

    Usage inside clean():
        df = analyze_and_impute(df, threshold=0.05, knn_neighbors=5)
    """
    print("\n" + "=" * 65)
    print("  MISSING DATA ANALYSIS & IMPUTATION")
    print("=" * 65)

    # Step 1: Find columns with missing values (threshold default = 0.0)
    missing_cols = get_high_missing_cols(df, threshold)
    all_missing  = list(missing_cols["numeric"].keys()) + list(missing_cols["categorical"].keys())

    if not all_missing:
        print("\nNo features exceed the missing threshold — skipping imputation.")
        return df

    # Step 2: MAR analysis
    mar_results = analyze_missingness(df, missing_cols)

    # Step 3: Impute numeric
    df = impute_numeric(df, list(missing_cols["numeric"].keys()), n_neighbors=knn_neighbors)

    # Step 4: Impute categorical
    df = impute_categorical(df, list(missing_cols["categorical"].keys()))

    # Step 5: Validate
    validate_imputation(df, all_missing)

    print("\n" + "=" * 65)

    return df
