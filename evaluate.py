"""
Train and evaluate WER prediction models using grouped cross-validation and
feature ablation.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                            confusion_matrix, f1_score, matthews_corrcoef,
                            precision_score, r2_score, recall_score,
                            roc_auc_score)
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence

from features import block_of, build_features, PROXY_ROLES


def make_model(name="hgb", seed=0):
    """
    Create a regression model.

    Args:
        name: Model name.
        seed: Random seed.

    Returns:
        Configured regression model.
    """
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_depth=3, max_iter=200, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=False, random_state=seed,
        )
    if name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(
            max_depth=3, n_estimators=200, learning_rate=0.05,
            min_child_weight=20, reg_lambda=1.0, subsample=0.8,
            colsample_bytree=0.8, objective="reg:absoluteerror",
            importance_type="gain", random_state=seed, n_jobs=-1,
        )
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"unknown model: {name}")


def select_blocks(X, names, blocks):
    """
    Select features belonging to the requested blocks.

    Args:
        X: Feature matrix.
        names: Feature names.
        blocks: Feature blocks to retain.

    Returns:
        Filtered feature matrix and selected feature names.
    """
    if not blocks:
        return np.zeros((len(X), 1)), []
    keep = [j for j, name in enumerate(names) if block_of(name) in blocks]
    return np.asarray(X)[:, keep], [names[j] for j in keep]


def splitter(groups, n_splits=5):
    """
    Create a grouped cross-validation splitter.

    Args:
        groups: Group labels.
        n_splits: Number of folds.

    Returns:
        Cross-validation splitter.
    """
    n_groups = len(set(groups))
    if n_splits >= n_groups:
        return LeaveOneGroupOut()
    return GroupKFold(n_splits=n_splits)


def cross_validate(rows, blocks=("proxy", "text"), target="wer",  # blocks=("proxy", "text", "extra")
                   model="hgb", n_splits=5, seed=0, roles=None):
    """
    Perform grouped cross-validation.

    Args:
        rows: Input records.
        blocks: Feature blocks to include.
        target: Prediction target.
        model: Regression model.
        n_splits: Number of folds.
        seed: Random seed.
        roles: Proxy ASR systems.

    Returns:
        Cross-validation predictions and evaluation data.
    """
    from features import PROXY_ROLES

    X, y_errors, y_wer, groups, names = build_features(
        rows, blocks=blocks, roles=roles or PROXY_ROLES
    )
    X, used_names = select_blocks(X, names, blocks)
    y_errors = np.asarray(y_errors, dtype=float)
    y_wer = np.asarray(y_wer, dtype=float)
    # groups = np.asarray(groups)
    cv_groups = np.array([r.get("group_id") or r["sample_id"] for r in rows])
    sample_ids = np.array([r["sample_id"] for r in rows])

    n_hyp = np.array([max(r.get("n_hyp_words", 0), 1) for r in rows], dtype=float)

    y = y_errors if target == "errors" else y_wer
    predictions = np.zeros(len(y), dtype=float)
    fold_scores = []

    # cv = splitter(groups, n_splits)
    cv = splitter(cv_groups, n_splits)
    for train_index, test_index in cv.split(X, y, cv_groups):
        estimator = make_model(model, seed)
        estimator.fit(X[train_index], y[train_index])
        fold_prediction = estimator.predict(X[test_index])
        predictions[test_index] = fold_prediction
        fold_scores.append(
            float(np.mean(np.abs(fold_prediction - y[test_index])))
        )

    if target == "errors":
        predicted_errors = np.clip(predictions, 0, None)
        predicted_wer = np.clip(predicted_errors / n_hyp, 0, 2.0)
    else:
        predicted_wer = np.clip(predictions, 0, 2.0)
        predicted_errors = predicted_wer * n_hyp

    return {
        "blocks": tuple(blocks), "target": target, "model": model,
        "feature_names": used_names, "n_features": len(used_names),
        "groups": sample_ids,
        "y_wer": y_wer, "pred_wer": predicted_wer,
        "y_errors": y_errors, "pred_errors": predicted_errors,
        "durations": np.array([r["duration"] for r in rows], dtype=float),
        "fold_mae": fold_scores,
        "n_folds": len(fold_scores),
    }


def segment_metrics(result):
    """
    Compute segment-level evaluation metrics.

    Args:
        result: Cross-validation results.

    Returns:
        Dictionary of evaluation metrics.
    """
    y, prediction = result["y_wer"], result["pred_wer"]
    return {
        "MAE_wer": float(np.mean(np.abs(prediction - y))),
        "RMSE_wer": float(np.sqrt(np.mean((prediction - y) ** 2))),
        # "MAE_errors": float(np.mean(np.abs(
        #     result["pred_errors"] - result["y_errors"]))),
        "pearson": float(stats.pearsonr(y, prediction)[0]),
        "spearman": float(stats.spearmanr(y, prediction)[0]),
        "fold_mae_std": float(np.std(result["fold_mae"])),
    }


def call_metrics(result):
    """
    Compute call-level evaluation metrics.

    Args:
        result: Cross-validation results.

    Returns:
        Call-level metrics and summary statistics.
    """
    rows = []
    for group in sorted(set(result["groups"])):
        mask = result["groups"] == group
        weights = result["durations"][mask]
        total = weights.sum()
        if total <= 0:
            continue
        true_wer = float(np.sum(result["y_wer"][mask] * weights) / total)
        estimated = float(np.sum(result["pred_wer"][mask] * weights) / total)
        rows.append({
            "call": group, "n_segments": int(mask.sum()),
            "wer_true": round(true_wer, 4),
            "wer_pred": round(estimated, 4),
            "abs_error": round(abs(estimated - true_wer), 4),
            "rel_error": round(abs(estimated - true_wer) / true_wer, 4)
            if true_wer > 0 else None,
        })

    relatives = [r["rel_error"] for r in rows if r["rel_error"] is not None]
    true_values = np.array([r["wer_true"] for r in rows])
    predicted_values = np.array([r["wer_pred"] for r in rows])
    summary = {
        "n_calls": len(rows),
        "WERR_mean": float(np.mean(relatives)) if relatives else None,
        "MAE_call": float(np.mean(np.abs(predicted_values - true_values))),
    }
    if len(rows) > 2:
        summary["spearman_call"] = float(
            stats.spearmanr(true_values, predicted_values)[0]
        )
    return rows, summary


def denominator_bias(rows):
    """
    Measure the difference between reference and hypothesis lengths.

    Args:
        rows: Input records.

    Returns:
        Length ratio statistics.
    """
    ratios = np.array([
        r["n_ref_words"] / max(r.get("n_hyp_words", 0), 1) for r in rows
    ], dtype=float)
    return {
        "ratio_mean": float(ratios.mean()),
        "ratio_median": float(np.median(ratios)),
        "ratio_p10": float(np.percentile(ratios, 10)),
        "ratio_p90": float(np.percentile(ratios, 90)),
        "pct_within_10pct": float(np.mean(np.abs(ratios - 1) < 0.1)), # Pct...
    }


ABLATION_PLAN = [
    ("baseline (mean)", (), "dummy"),
    ("text only", ("text",), "hgb"),
    ("proxy only", ("proxy",), "hgb"),
    ("(proxy + text) | hgb ", ("proxy", "text"), "hgb"),
    ("(proxy + text) | xgb", ("proxy", "text"), "xgb"),
    ("(proxy + text) | ridge", ("proxy", "text"), "ridge"),
]

# ABLATION_PLAN = [
#     ("baseline (mean)", (), "dummy"),
#     ("text only ", ("text",), "hgb"),
#     ("proxy only", ("proxy",), "hgb"),
#     ("(proxy + text) | hgb ", ("proxy", "text"), "hgb"),
#     ("(proxy + text) | xgb", ("proxy", "text"), "xgb"),
#     ("(proxy + text) | ridge", ("proxy", "text"), "ridge"),
#     ("(proxy + text + extra) | hgb", ("proxy", "text", "extra"), "hgb"),
#     ("(proxy + text + extra) | xgb", ("proxy", "text", "extra"), "xgb"),
#     ("(proxy + text + extra) | ridge", ("proxy", "text", "extra"), "ridge"),
# ]


def ablation(rows, target="wer", n_splits=5, seed=0, plan=None):
    """
    Evaluate different feature configurations.

    Args:
        rows: Input records.
        target: Prediction target.
        n_splits: Number of folds.
        seed: Random seed.
        plan: Ablation configurations.

    Returns:
        Ablation results.
    """
    table = []
    for label, blocks, model in (plan or ABLATION_PLAN):
        result = cross_validate(rows, blocks=blocks, target=target,
                                model=model, n_splits=n_splits, seed=seed)
        metrics = segment_metrics(result)
        _, call_summary = call_metrics(result)
        table.append({
            "config": label,
            "n_feat": result["n_features"],
            "MAE_wer": round(metrics["MAE_wer"], 4),
            "RMSE_wer": round(metrics["RMSE_wer"], 4),
            "pearson": round(metrics["pearson"], 4),
            "spearman": round(metrics["spearman"], 4),
            "WERR": round(call_summary["WERR_mean"], 4) # per-call metric (th
            if call_summary["WERR_mean"] is not None else None,
        })
    return table


def single_proxy_ablation(rows, target="wer", model="hgb", n_splits=5):
    """
    Evaluate the contribution of each proxy ASR system.

    Args:
        rows: Input records.
        target: Prediction target.
        model: Regression model.
        n_splits: Number of folds.

    Returns:
        Evaluation results for each proxy configuration.
    """
    from features import PROXY_ROLES

    table = []
    for roles in [(PROXY_ROLES[0],), (PROXY_ROLES[1],), PROXY_ROLES]:
        result = cross_validate(rows, blocks=("proxy", "text"), target=target,
                                model=model, n_splits=n_splits, roles=roles)
        metrics = segment_metrics(result)
        _, call_summary = call_metrics(result)
        table.append({
            "proxies": "+".join(roles),
            "MAE_wer": round(metrics["MAE_wer"], 4),
            "spearman": round(metrics["spearman"], 4),
            "WERR": round(call_summary["WERR_mean"], 4)
            if call_summary["WERR_mean"] is not None else None,
        })
    return table


def permutation_importance(rows, blocks=("proxy", "text"), target="wer", model="hgb",
                           n_splits=5, n_repeats=5, seed=0):
    """
    Estimate feature importance using permutation.

    Args:
        rows: Input records.
        blocks: Feature blocks to include.
        target: Prediction target.
        model: Regression model.
        n_splits: Number of folds.
        n_repeats: Number of permutations.
        seed: Random seed.

    Returns:
        Baseline score and feature importance values.
    """
    rng = np.random.default_rng(seed)
    reference = cross_validate(rows, blocks, target, model, n_splits, seed)
    base_mae = segment_metrics(reference)["MAE_wer"]

    from features import PROXY_ROLES

    X, y_errors, y_wer, groups, names = build_features(
        rows, blocks=blocks, roles=PROXY_ROLES
    )
    X, used_names = select_blocks(X, names, blocks)
    y = np.asarray(y_errors if target == "errors" else y_wer, dtype=float)
    groups = np.asarray(groups)
    n_hyp = np.array([max(r.get("n_hyp_words", 0), 1) for r in rows], dtype=float)
    cv = splitter(groups, n_splits)

    scores = []
    for j, name in enumerate(used_names):
        deltas = []
        for _ in range(n_repeats):
            predictions = np.zeros(len(y))
            for train_index, test_index in cv.split(X, y, groups):
                estimator = make_model(model, seed)
                estimator.fit(X[train_index], y[train_index])
                shuffled = X[test_index].copy()
                shuffled[:, j] = rng.permutation(shuffled[:, j])
                predictions[test_index] = estimator.predict(shuffled)
            if target == "errors":
                predicted_wer = np.clip(np.clip(predictions, 0, None) / n_hyp, 0, 2)
            else:
                predicted_wer = np.clip(predictions, 0, 2)
            deltas.append(float(np.mean(np.abs(predicted_wer - y_wer))) - base_mae)
        scores.append((name, float(np.mean(deltas)), float(np.std(deltas))))

    scores.sort(key=lambda item: -item[1])
    return base_mae, scores


def _matrix(rows, blocks, target, roles=PROXY_ROLES):
    """
    Build the feature matrix used for model analysis.

    Args:
        rows: Input records.
        blocks: Feature blocks to include.
        target: Prediction target.
        roles: Proxy ASR systems.

    Returns:
        Feature matrix, target vector, groups, and feature names.
    """
    X, y_errors, y_wer, groups, names = build_features(rows, roles=roles)
    X, used = select_blocks(X, names, blocks)
    y = np.asarray(y_errors if target == "errors" else y_wer, dtype=float)
    return np.asarray(X), y, np.asarray(groups), used


def ridge_coefficients(rows, blocks=("proxy", "text"), target="wer", n_splits=5):
    """
    Estimate Ridge regression coefficients across folds.

    Args:
        rows: Input records.
        blocks: Feature blocks to include.
        target: Prediction target.
        n_splits: Number of folds.

    Returns:
        Averaged feature coefficients.
    """
    X, y, groups, names = _matrix(rows, blocks, target)
    cv = splitter(groups, n_splits)

    coefficients = []
    for train_index, _ in cv.split(X, y, groups):
        model = make_model("ridge").fit(X[train_index], y[train_index])
        coefficients.append(model[-1].coef_)      # [-1] = Ridge, [0] = scaler
    coefficients = np.array(coefficients)

    table = [
        {"feature": name,
         "coef": round(float(coefficients[:, j].mean()), 4),
         "std": round(float(coefficients[:, j].std()), 4),
         "stable": bool(abs(coefficients[:, j].mean()) > coefficients[:, j].std())}
        for j, name in enumerate(names)
    ]
    table.sort(key=lambda item: -abs(item["coef"]))
    return table


def partial_dependence_curve(rows, feature, blocks=("proxy", "text"), model_name="hgb",
                             target="wer", n_points=10, seed=0):
    """
    Compute the partial dependence of a feature.

    Args:
        rows: Input records.
        feature: Feature name.
        blocks: Feature blocks to include.
        target: Prediction target.
        model_name: Regression model.
        n_points: Number of evaluation points.
        seed: Random seed.

    Returns:
        Partial dependence values.
    """
    from sklearn.inspection import partial_dependence

    X, y, groups, names = _matrix(rows, blocks, target)
    if feature not in names:
        raise ValueError(f"unknown feature: {feature}")

    model = make_model(model_name, seed).fit(X, y)
    result = partial_dependence(model, X, [names.index(feature)],
                                grid_resolution=n_points, kind="average")

    return [{"value": round(float(v), 4), "predicted": round(float(p), 4)}
            for v, p in zip(result["grid_values"][0], result["average"][0])]


def xgb_importances(rows, blocks=("proxy", "text"), target="wer", seed=0):
    """
    Compute feature importance using XGBoost.

    Args:
        rows: Input records.
        blocks: Feature blocks to include.
        target: Prediction target.
        seed: Random seed.

    Returns:
        Feature importance scores.
    """
    X, y, groups, names = _matrix(rows, blocks, target)
    model = make_model("xgb", seed).fit(X, y)
    table = sorted(zip(names, model.feature_importances_),
                   key=lambda item: -item[1])
    return [{"feature": n, "gain": round(float(g), 4)} for n, g in table]


# -----------------------------------------------------------------------------
# Extended metrics
# -----------------------------------------------------------------------------

BAD_THRESHOLD = 0.25


def regression_metrics(y_true, y_pred, mape_floor=0.02):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    usable = y_true > mape_floor
    mape = (float(np.mean(np.abs(residual[usable] / y_true[usable])))) \
        if usable.any() else None

    return {
        "n": len(y_true),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": mape,
        "MAPE_excluded": int((~usable).sum()),
        "bias": float(np.mean(residual)),
        "pearson": float(stats.pearsonr(y_true, y_pred)[0]),
        "spearman": float(stats.spearmanr(y_true, y_pred)[0]),
        "kendall": float(stats.kendalltau(y_true, y_pred)[0]),
    }


def binary_metrics(y_true, y_pred, threshold=BAD_THRESHOLD):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    true_bad = (y_true > threshold).astype(int)
    pred_bad = (y_pred > threshold).astype(int)

    report = {
        "threshold": threshold,
        "n": len(y_true),
        "n_bad_true": int(true_bad.sum()),
        "prevalence": float(true_bad.mean()),
        "n_bad_pred": int(pred_bad.sum()),
    }

    if 0 < true_bad.sum() < len(true_bad):
        report.update({
            "ROC_AUC": float(roc_auc_score(true_bad, y_pred)),
            "PR_AUC": float(average_precision_score(true_bad, y_pred)),
            "precision": float(precision_score(true_bad, pred_bad, zero_division=0)),
            "recall": float(recall_score(true_bad, pred_bad, zero_division=0)),
            "F1": float(f1_score(true_bad, pred_bad, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(true_bad, pred_bad)),
            "MCC": float(matthews_corrcoef(true_bad, pred_bad)),
        })
    else:
        report["note"] = "single class at this threshold - ranking metrics undefined"

    matrix = confusion_matrix(true_bad, pred_bad, labels=[0, 1])
    return report, matrix


def precision_at_k(y_true, y_pred, k, threshold=BAD_THRESHOLD):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    true_bad = (y_true > threshold).astype(int)
    k = min(int(k), len(y_true))
    if k == 0:
        return None

    top = np.argsort(-y_pred)[:k]
    precision = float(true_bad[top].mean())
    prevalence = float(true_bad.mean())

    return {
        "k": k,
        "precision_at_k": precision,
        "prevalence": prevalence,
        "lift": float(precision / prevalence) if prevalence > 0 else None,
    }


def full_report(y_true, y_pred, threshold=BAD_THRESHOLD):

    binary, matrix = binary_metrics(y_true, y_pred, threshold)
    return {
        "regression": regression_metrics(y_true, y_pred),
        "binary": binary,
        "confusion_matrix": matrix,
    }