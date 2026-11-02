"""llmdiff.analytics — statistical analysis of evaluation runs."""

from __future__ import annotations

from llmdiff.analytics.anomalies import (
    detect_outliers,
    distance_outliers,
    iqr_fences,
    iqr_outliers,
    mad_outliers,
    modified_zscore,
    zscore,
    zscore_outliers,
)
from llmdiff.analytics.clustering import (
    agglomerative,
    cluster_regressions,
    euclidean,
    kmeans,
    kmeans_pp_seed,
    silhouette,
)
from llmdiff.analytics.correlations import (
    correlation_matrix,
    covariance_matrix,
    pca,
    pearson,
    rank,
    redundant_dimensions,
    spearman,
)
from llmdiff.analytics.db import (
    RunRow,
    aggregate_by_model,
    aggregate_by_tag,
    load_result,
    query_runs,
    run_history,
)
from llmdiff.analytics.distributions import (
    fit_beta,
    fit_lognormal,
    fit_normal,
    ks_statistic,
    ks_test_pvalue,
    log_likelihood,
    qq_data,
)
from llmdiff.analytics.power import (
    minimum_detectable_effect,
    normal_cdf,
    normal_ppf,
    power_for_win_rate,
    sample_size_for_win_rate,
)
from llmdiff.analytics.significance import (
    benjamini_hochberg,
    bonferroni,
    cohens_d,
    correct_pvalues,
    holm,
    paired_t_test,
    sign_test,
    wilcoxon_signed_rank,
)
from llmdiff.analytics.stats import (
    bootstrap_ci,
    describe,
    histogram,
    mean,
    median,
    min_max,
    percentile,
    std,
    variance,
)
from llmdiff.analytics.trends import (
    cusum_change_point,
    forecast_linear,
    linear_regression,
    moving_average,
    r_squared,
    run_series_from_results,
    slope,
)

__all__ = [
    "mean", "variance", "std", "median", "percentile", "min_max",
    "describe", "bootstrap_ci", "histogram",
    "sign_test", "wilcoxon_signed_rank", "paired_t_test", "cohens_d",
    "bonferroni", "holm", "benjamini_hochberg", "correct_pvalues",
    "linear_regression", "slope", "r_squared", "cusum_change_point",
    "forecast_linear", "moving_average", "run_series_from_results",
    "pearson", "spearman", "covariance_matrix", "correlation_matrix",
    "rank", "pca", "redundant_dimensions",
    "euclidean", "kmeans", "kmeans_pp_seed", "agglomerative", "silhouette",
    "cluster_regressions",
    "zscore", "zscore_outliers", "modified_zscore", "mad_outliers",
    "iqr_outliers", "iqr_fences", "distance_outliers", "detect_outliers",
    "fit_normal", "fit_lognormal", "fit_beta", "ks_statistic",
    "ks_test_pvalue", "qq_data", "log_likelihood",
    "normal_cdf", "normal_ppf", "sample_size_for_win_rate",
    "power_for_win_rate", "minimum_detectable_effect",
    "RunRow", "query_runs", "run_history", "aggregate_by_model",
    "aggregate_by_tag", "load_result",
]
