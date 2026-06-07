"""
Missing Data Analysis & Imputation
====================================
Analyzes missingness patterns for all features with > 5% missing values,
determines whether missingness is MCAR or MAR via correlation analysis,
and applies the appropriate imputation strategy per feature.

Imputation strategy:
  - Numeric features:     KNN imputation using each column's top-5 correlates
                          from MAR analysis as the neighbor-computation context.
  - Categorical features: Same — label-encoded then KNN-imputed using each
                          column's top-5 correlates; decoded back to original
                          categories after imputation.

MAR analysis:
  - Missingness indicator for each high-missing column is correlated against
    ALL other features in the DataFrame (not just a demographic subset).
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

    # Drop columns with > 60% missing
    cols_to_drop = missing_pct[missing_pct > 0.60].index.tolist()
    if cols_to_drop:
        print(f"\n  Dropping {len(cols_to_drop)} columns with > 60% missing: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
        high_missing = high_missing.drop(labels=[c for c in cols_to_drop if c in high_missing.index])

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


# ── 2. MAR correlation analysis ───────────────────────────────────────────────
def analyze_missingness(df: pd.DataFrame, missing_cols: dict) -> tuple[dict, dict]:
    """
    For each column with high missingness, creates a binary missingness
    indicator and correlates it against ALL other features in the DataFrame.

    Categorical reference columns are label-encoded before correlation so
    every feature can participate regardless of dtype.

    Returns:
        mar_results        : {col: "MAR" | "MCAR" | "NONE"}
        correlation_details: {col: pd.Series of |r| values sorted descending}
                             Used by impute_* functions to select top-5 context
                             features for each column's KNN imputer.

    MAR threshold: any absolute correlation > 0.1 with any other variable.
    """
    all_high_missing = list(missing_cols["numeric"].keys()) + list(missing_cols["categorical"].keys())

    if not all_high_missing:
        print("\nNo features exceed the missing threshold — skipping analysis.")
        return {}, {}

    # Build a fully numeric reference DataFrame from ALL other columns.
    # Categorical columns are label-encoded so they can be correlated.
    ID_target_cols = {'SEQN', 'hypertension_risk', 'participant_id'}
    reference_cols = [c for c in df.columns if c not in all_high_missing and c not in ID_target_cols]
    df_ref = df[reference_cols].copy()

    for col in df_ref.columns:
        if not pd.api.types.is_numeric_dtype(df_ref[col]):
            df_ref[col] = LabelEncoder().fit_transform(df_ref[col].astype(str))
        else:
            df_ref[col] = pd.to_numeric(df_ref[col], errors='coerce')

    print(f"  Reference pool: {len(reference_cols)} features (all columns not in the high-missing set)")
    print(f"\n── MAR Correlation Analysis ─────────────────────────────────────────")
    print(f"  Threshold: |correlation| > 0.1 with any feature → MAR")
    print(f"  Otherwise → MCAR\n")

    mar_results        = {}
    correlation_details = {}

    for col in all_high_missing:
        # Binary missingness indicator: 1 = missing, 0 = observed
        missing_indicator = df[col].isna().astype(int)

        if missing_indicator.sum() == 0:
            mar_results[col]        = "NONE"
            correlation_details[col] = pd.Series(dtype=float)
            continue

        # Correlate missingness indicator against every reference feature
        corr_series   = df_ref.corrwith(missing_indicator).abs().dropna().sort_values(ascending=False)
        max_corr      = corr_series.max() if not corr_series.empty else 0.0
        top_predictor = corr_series.idxmax() if not corr_series.empty else "none"
        classification = "MAR" if max_corr > 0.1 else "MCAR"

        mar_results[col]         = classification
        correlation_details[col] = corr_series

        # Print top 5 correlated features
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
    print(f"  MAR: {mar_count} features")
    print(f"  MCAR:   {mcar_count} features")

    return mar_results, correlation_details


# ── 3. Impute numeric features ────────────────────────────────────────────────
def impute_numeric(df: pd.DataFrame, numeric_missing_cols: list,
                   correlation_details: dict,
                   n_neighbors: int = 5,
                   top_n_context: int = 5) -> pd.DataFrame:
    """
    Imputes each numeric column with KNN using only that column's top-N
    correlated features (from MAR analysis) as the neighbor-computation context.

    For each target column the feature matrix passed to KNNImputer is:
        [target_col] + top_n_context columns from correlation_details[target_col]

    If fewer than top_n_context correlates are available the imputer uses
    whatever is present — KNNImputer handles residual missingness in context
    columns internally.

    Binary columns (0/1) are rounded back to integers after imputation.
    """
    if not numeric_missing_cols:
        print("\n  No numeric columns to impute.")
        return df

    print(f"\n── Numeric Imputation (KNN, k={n_neighbors}, top-{top_n_context} context) ──────────────")

    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    binary_cols      = {col for col in all_numeric_cols if df[col].dropna().isin([0, 1]).all()}

    for col in numeric_missing_cols:
        if col not in df.columns:
            continue

        n_missing = df[col].isna().sum()
        print(f"\n  {col:<45} missing before: {n_missing:>5} ({n_missing/len(df)*100:.1f}%)")

        # Select top-N context features from MAR correlation details
        corr_series   = correlation_details.get(col, pd.Series(dtype=float))
        context_cols  = [c for c in corr_series.index[:top_n_context] if c in df.columns]
        feature_cols  = [col] + [c for c in context_cols if c != col]

        print(f"    Context features: {context_cols}")

        # Build a numeric-only sub-DataFrame for the imputer.
        # Label-encode any categorical context columns so KNNImputer can use them.
        df_context = pd.DataFrame(index=df.index)
        encoders   = {}
        for fc in feature_cols:
            if pd.api.types.is_numeric_dtype(df[fc]):
                df_context[fc] = df[fc]
            else:
                le = LabelEncoder()
                # fit only on non-null values; map NaN → NaN after encoding
                non_null = df[fc].dropna()
                le.fit(non_null.astype(str))
                encoded = df[fc].map(lambda v: le.transform([str(v)])[0]
                                     if pd.notna(v) else np.nan)
                df_context[fc] = encoded
                encoders[fc]   = le

        imputer        = KNNImputer(n_neighbors=n_neighbors)
        imputed_array  = imputer.fit_transform(df_context)
        df_imputed_ctx = pd.DataFrame(imputed_array, columns=feature_cols, index=df.index)

        # Write only the target column back — context columns are unchanged
        df[col] = df_imputed_ctx[col]

        if col in binary_cols:
            df[col] = df[col].round().astype("Int64")

        print(f"    {col:<45} missing after:  {df[col].isna().sum():>5}")

    zeroed = zero_near_zero_values(df)
    if zeroed > 0:
        print(f"\n  Zeroed {zeroed:,} tiny numeric values to 0")

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
def impute_categorical(
    df: pd.DataFrame,
    categorical_missing_cols: list,
    correlation_details: dict,
    top_n_context: int = 5,
    **kwargs,                     # absorbs unused n_neighbors so call-sites need no change
) -> pd.DataFrame:
    """
    Imputes each categorical column with grouped mode, where the groups are
    defined by that column's top-N correlated features from MAR analysis.

    Approach per column:
      1. Pull top-N correlates from correlation_details as groupby keys.
         Only features that are already fully observed (no NaNs) are used as
         groupby keys to avoid pandas groupby dropping rows with NaN keys.
      2. Compute the mode of the target column within each group cell.
      3. Fill missing rows with their group's mode.
      4. Fall back to the global mode for any rows whose group had no observed
         values (e.g., a rare combination of group keys).
      5. Restore the original dtype (including pandas Categorical).
    """
    if not categorical_missing_cols:
        print("\n  No categorical columns to impute.")
        return df

    print(f"\n── Categorical Imputation (grouped mode, top-{top_n_context} MAR context) ──────────────")

    def _mode(series: pd.Series):
        m = series.dropna().mode()
        return m.iloc[0] if not m.empty else np.nan

    for col in categorical_missing_cols:
        if col not in df.columns:
            continue

        n_missing      = df[col].isna().sum()
        original_dtype = df[col].dtype
        missing_mask   = df[col].isna()

        print(f"\n  {col:<45} missing before: {n_missing:>5} ({n_missing/len(df)*100:.1f}%)")

        # Top-N correlates from MAR analysis, restricted to fully-observed
        # columns so groupby keys are always clean.
        corr_series   = correlation_details.get(col, pd.Series(dtype=float))
        candidate_ctx = [c for c in corr_series.index[:top_n_context]
                         if c in df.columns and c != col]
        group_cols    = [c for c in candidate_ctx if df[c].isna().sum() == 0]

        # If all candidates have some missingness, fall back to the one with
        # the fewest missing values so we still use MAR context.
        if not group_cols and candidate_ctx:
            group_cols = [min(candidate_ctx, key=lambda c: df[c].isna().sum())]

        print(f"    Groupby features: {group_cols}")

        if group_cols:
            group_modes = df.groupby(group_cols)[col].transform(_mode)
            df.loc[missing_mask, col] = group_modes[missing_mask]

        # Global mode fallback for any rows still missing
        still_missing = df[col].isna()
        if still_missing.any():
            global_mode = _mode(df[col])
            df.loc[still_missing, col] = global_mode
            print(f"    Global mode fallback used for {still_missing.sum()} rows")

        if str(original_dtype) == "category":
            df[col] = df[col].astype("category")

        print(f"    {col:<45} missing after:  {df[col].isna().sum():>5}")

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
                       knn_neighbors: int = 5,
                       top_n_context: int = 5) -> pd.DataFrame:
    """
    Full pipeline:
      1. Identify columns with > threshold% missing
      2. Run MAR correlation analysis against ALL other features
      3. Impute numeric columns with per-column KNN (top-N context features)
      4. Impute categorical columns with per-column KNN (top-N context features)
      5. Validate results

    Returns the imputed DataFrame.

    Usage inside clean():
        df = analyze_and_impute(df, threshold=0.05, knn_neighbors=5, top_n_context=5)
    """
    print("\n" + "=" * 65)
    print("  MISSING DATA ANALYSIS & IMPUTATION")
    print("=" * 65)

    # Step 1: Find columns with missing values
    missing_cols = get_high_missing_cols(df, threshold)
    all_missing  = list(missing_cols["numeric"].keys()) + list(missing_cols["categorical"].keys())

    if not all_missing:
        print("\nNo features exceed the missing threshold — skipping imputation.")
        return df

    # Step 2: MAR analysis — returns both classification and per-column correlations
    mar_results, correlation_details = analyze_missingness(df, missing_cols)

    # Step 3: Impute numeric using top-N correlated context features
    df = impute_numeric(
        df,
        list(missing_cols["numeric"].keys()),
        correlation_details=correlation_details,
        n_neighbors=knn_neighbors,
        top_n_context=top_n_context,
    )

    # Step 4: Impute categorical using top-N correlated context features
    df = impute_categorical(
        df,
        list(missing_cols["categorical"].keys()),
        correlation_details=correlation_details,
        n_neighbors=knn_neighbors,
        top_n_context=top_n_context,
    )

    # Step 5: Validate
    validate_imputation(df, all_missing)

    print("\n" + "=" * 65)

    return df
