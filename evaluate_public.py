"""
Evaluation protocols and baselines for the public pool.

Everything here answers one question: does the estimator still work when the
test data was not drawn from the training distribution? Four protocols, each
one holding out a different axis:

    group  in-distribution, grouped k-fold (leakage-safe within the pool)
    lodo   leave-one-dataset-out, the protocol of Waheed et al. (ACL 2025)
    lolo   leave-one-language-out
    loco   leave-one-condition-out, does the model extrapolate to an unseen
           acoustic degradation

Baselines, reported next to every model so that a score can be read:

    mean      predict the training mean (a positive R2 must beat this)
    pwer      use the raw proxy disagreement as the WER estimate. This is the
              "W PROXY" baseline of Waheed et al., table 4: their regression
              beats it on 7 datasets out of 8
    pwer_lin  univariate linear regression on the same quantity
    ridge     standardized ridge, the model whose coefficients are transferred

Metrics reuse evaluate.regression_metrics (MAE, RMSE, R2, bias, Pearson,
Spearman). Two additions:
  - corpus_wer_gap, the difference between the aggregate WER announced by the
    estimator and the true one. Waheed et al. compare WER and aWER at dataset
    level, eWER3 (Chowdhury and Ali, 2023) reports the same aggregate. It is a
    bias, not a dispersion: segment errors cancel in the aggregate, so it is not
    redundant with MAE.
  - a cluster bootstrap over groups, giving confidence intervals that account
    for the fact that segments within a group are not independent.
"""

import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import make_model, regression_metrics
from features import PROXY_ROLES, build_features


WER_CLIP = (0.0, 2.0)

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
LASSO_ALPHAS = (0.0001, 0.001, 0.01, 0.1)

# Kept small on purpose: the grid is searched inside every outer fold, so its
# size multiplies the runtime of every protocol.
HGB_GRID = {
    "max_depth": [3, 5, None],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_iter": [200, 400],
    "min_samples_leaf": [10, 20],
}

XGB_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [200, 400],
    "min_child_weight": [5, 20],
}

# Feature used by the "pwer" baseline. pwer_mean averages the disagreement with
# both proxies; Waheed et al. use a single proxy, the best-ranked model other
# than the target, so set this to "pwer_proxy_a" for strict parity with their
# W PROXY baseline.
PROXY_BASELINE_FEATURE = "pwer_mean"

# Features that are ratios or rates, hence comparable across corpora with
# different segment lengths. Counts and durations are excluded from the
# transferable set: their scale is corpus-specific, and x_is_operator is
# constant on public data.
SCALE_FREE_PREFIXES = ("pwer", "pcer", "psub", "pdel", "pins", "plen")
SCALE_FREE_TEXT = ("gzip_ratio", "type_token_ratio", "repeated_word_ratio",
                   "chars_per_word", "words_per_second")

MODELS = ("mean", "pwer", "ridge", "lasso", "hgb", "xgb")


def is_transferable(name):
    """
    Whether a feature keeps its meaning across corpora of different lengths.

    Args:
        name: Feature name.

    Returns:
        True when the feature is scale free.
    """
    return name.startswith(SCALE_FREE_PREFIXES) or name in SCALE_FREE_TEXT


def design(rows, blocks=("proxy", "text"), roles=PROXY_ROLES,
           feature_set="transferable"):
    """
    Build the design matrix and the metadata arrays used by every protocol.

    Args:
        rows: Labelled records.
        blocks: Feature blocks to include.
        roles: Proxy ASR systems.
        feature_set: "transferable" or "full".

    Returns:
        Feature matrix, feature names and metadata arrays.
    """
    X, _, y_wer, _, names = build_features(rows, blocks=blocks, roles=roles)
    X = np.asarray(X, dtype=float)

    if feature_set == "transferable":
        keep = [i for i, name in enumerate(names) if is_transferable(name)]
    elif feature_set == "full":
        keep = list(range(len(names)))
    else:
        raise ValueError(f"unknown feature_set {feature_set}")

    meta = {
        "y": np.asarray(y_wer, dtype=float),
        "cv_group": np.array([r.get("group_id") or r["sample_id"] for r in rows]),
        "corpus": np.array([r.get("corpus", "internal") for r in rows]),
        "lang": np.array([r["lang"] for r in rows]),
        "condition": np.array([r.get("condition", "clean") for r in rows]),
        "n_ref_words": np.array([r["n_ref_words"] for r in rows], dtype=float),
        "n_hyp_words": np.array([max(r.get("n_hyp_words", 0), 1) for r in rows],
                                dtype=float),
        "label_errors": np.array([r["label_errors"] for r in rows], dtype=float),
    }
    return X[:, keep], [names[i] for i in keep], meta


def proxy_column(names):
    """
    Locate the column holding the raw proxy disagreement.

    Args:
        names: Feature names.

    Returns:
        Column index of pwer_mean, or of the first pwer feature available.
    """
    if PROXY_BASELINE_FEATURE in names:
        return names.index(PROXY_BASELINE_FEATURE)
    for index, name in enumerate(names):
        if name.startswith("pwer_"):
            return index
    raise ValueError("no proxy feature in the design matrix")


def splits(protocol, meta, n_splits=5):
    """
    Yield the train and test indices of a protocol.

    Args:
        protocol: One of "group", "lodo", "lolo", "loco".
        meta: Metadata arrays from design().
        n_splits: Number of folds for the "group" protocol.

    Returns:
        List of (fold name, train indices, test indices).
    """
    if protocol == "group":
        groups = meta["cv_group"]
        n_groups = len(set(groups))
        splitter = GroupKFold(n_splits=min(n_splits, n_groups))
        dummy = np.zeros(len(groups))
        return [(f"fold{i}", train, test) for i, (train, test)
                in enumerate(splitter.split(dummy, dummy, groups))]

    key = {"lodo": "corpus", "lolo": "lang", "loco": "condition"}.get(protocol)
    if key is None:
        raise ValueError(f"unknown protocol {protocol}")

    values = meta[key]
    folds = []
    for held in sorted(set(values)):
        test = np.where(values == held)[0]
        train = np.where(values != held)[0]
        if train.size and test.size:
            folds.append((f"{key}={held}", train, test))
    return folds


def build_estimator(model, seed=0):
    """
    Create an estimator and its hyperparameter grid.

    Linear models are standardized because their penalty is scale sensitive;
    tree ensembles are not, so they are used as returned by evaluate.make_model
    and stay identical to what the internal pipeline already trains.

    Args:
        model: Model name from MODELS.
        seed: Random seed.

    Returns:
        Estimator and grid, the grid being None when there is nothing to tune.
    """
    if model == "ridge":
        return (make_pipeline(StandardScaler(), Ridge()),
                {"ridge__alpha": list(RIDGE_ALPHAS)})
    if model == "lasso":
        return (make_pipeline(StandardScaler(), Lasso(max_iter=20000)),
                {"lasso__alpha": list(LASSO_ALPHAS)})
    if model == "hgb":
        return make_model("hgb", seed), dict(HGB_GRID)
    if model == "xgb":
        return make_model("xgb", seed), dict(XGB_GRID)
    raise ValueError(f"unknown model {model}")


def _tuned_fit(model, X, y, train, groups, seed):
    """
    Fit an estimator, tuning it with grouped inner folds when it has a grid.

    Args:
        model: Model name from MODELS.
        X: Feature matrix.
        y: Target vector.
        train: Training indices.
        groups: Cross-validation groups.
        seed: Random seed.

    Returns:
        Fitted estimator.
    """
    estimator, grid = build_estimator(model, seed)
    if grid is None:
        return estimator.fit(X[train], y[train])

    train_groups = groups[train] if groups is not None else None
    if train_groups is not None and len(set(train_groups)) >= 3:
        search = GridSearchCV(estimator, grid, cv=GroupKFold(3),
                              scoring="neg_mean_absolute_error")
        search.fit(X[train], y[train], groups=train_groups)
    else:
        search = GridSearchCV(estimator, grid,
                              cv=KFold(3, shuffle=True, random_state=seed),
                              scoring="neg_mean_absolute_error")
        search.fit(X[train], y[train])
    return search


def fit_predict(model, X, y, train, test, groups=None, seed=0):
    """
    Fit one model on a training fold and predict the test fold.

    Args:
        model: Model name from MODELS.
        X: Feature matrix.
        y: Target vector.
        train: Training indices.
        test: Test indices.
        groups: Cross-validation groups, used to tune without leakage.
        seed: Random seed.

    Returns:
        Predictions on the test fold.
    """
    if model == "mean":
        return np.full(test.size, float(y[train].mean()))

    if model == "pwer":
        return X[test, fit_predict.proxy_index]

    return _tuned_fit(model, X, y, train, groups, seed).predict(X[test])


fit_predict.proxy_index = 0  # set by run_protocol before use


def corpus_wer_gap(y_pred, meta, index=None):
    """
    Compare the aggregate WER announced by the estimator with the true one.

    The estimated corpus WER uses hypothesis words as denominator, since
    references are unknown at inference time; the true one uses reference words.

    Args:
        y_pred: Predicted segment WER.
        meta: Metadata arrays from design().
        index: Optional subset of indices.

    Returns:
        Estimated WER, true WER and their absolute difference.
    """
    index = np.arange(len(y_pred)) if index is None else index
    hyp_words = meta["n_hyp_words"][index]
    estimated = float(np.sum(y_pred[index] * hyp_words) / np.sum(hyp_words))
    true = float(np.sum(meta["label_errors"][index])
                 / np.sum(meta["n_ref_words"][index]))
    return {
        "wer_corpus_estimated": round(estimated, 4),
        "wer_corpus_true": round(true, 4),
        "wer_corpus_gap": round(abs(estimated - true), 4),
    }


def run_protocol(rows, protocol="group", models=MODELS, blocks=("proxy", "text"),
                 roles=PROXY_ROLES, feature_set="transferable", n_splits=5):
    """
    Run one protocol for every model and collect out-of-fold predictions.

    Args:
        rows: Labelled records.
        protocol: One of "group", "lodo", "lolo", "loco".
        models: Models to compare.
        blocks: Feature blocks to include.
        roles: Proxy ASR systems.
        feature_set: "transferable" or "full".
        n_splits: Number of folds for the "group" protocol.

    Returns:
        Out-of-fold predictions, per-fold scores and design metadata.
    """
    X, names, meta = design(rows, blocks, roles, feature_set)
    fit_predict.proxy_index = proxy_column(names)
    folds = splits(protocol, meta, n_splits)
    y = meta["y"]

    predictions = {model: np.zeros(len(y), dtype=float) for model in models}
    fold_scores = []

    for fold_name, train, test in folds:
        entry = {"fold": fold_name, "n_train": int(train.size),
                 "n_test": int(test.size),
                 "wer_std_test": round(float(y[test].std()), 4)}
        for model in models:
            prediction = np.clip(
                fit_predict(model, X, y, train, test, groups=meta["cv_group"]),
                *WER_CLIP,
            )
            predictions[model][test] = prediction
            entry[f"MAE_{model}"] = round(
                float(np.mean(np.abs(prediction - y[test]))), 4)
        fold_scores.append(entry)

    return {
        "protocol": protocol,
        "feature_set": feature_set,
        "feature_names": names,
        "n_features": len(names),
        "meta": meta,
        "predictions": predictions,
        "folds": fold_scores,
    }


def cluster_bootstrap(y_true, y_pred, groups, n_boot=1000, seed=0):
    """
    Confidence interval for the MAE, resampling groups rather than segments.

    Segments of one group share a speaker and a channel, so they are not
    independent; resampling groups keeps that dependence intact.

    Args:
        y_true: True WER.
        y_pred: Predicted WER.
        groups: Group labels.
        n_boot: Number of bootstrap replicates.
        seed: Random seed.

    Returns:
        Point estimate and 95 percent percentile interval.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_of = {g: np.where(groups == g)[0] for g in unique}

    values = []
    for _ in range(n_boot):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        index = np.concatenate([index_of[g] for g in drawn])
        values.append(float(np.mean(np.abs(y_pred[index] - y_true[index]))))

    return {
        "MAE": round(float(np.mean(np.abs(y_pred - y_true))), 4),
        "ci_low": round(float(np.percentile(values, 2.5)), 4),
        "ci_high": round(float(np.percentile(values, 97.5)), 4),
        "n_groups": int(unique.size),
    }


def paired_bootstrap(y_true, pred_a, pred_b, groups, n_boot=1000, seed=0):
    """
    Confidence interval for the MAE difference between two models.

    Comparing two separate confidence intervals is the wrong test: they can
    overlap while the paired difference is consistently in favour of one model,
    because both are evaluated on the same segments and their errors move
    together. Resampling the same groups for both models keeps that pairing.

    Args:
        y_true: True WER.
        pred_a: Predictions of the reference model, usually the baseline.
        pred_b: Predictions of the challenger.
        groups: Group labels.
        n_boot: Number of bootstrap replicates.
        seed: Random seed.

    Returns:
        Observed difference MAE(a) - MAE(b) and its 95 percent interval.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_of = {g: np.where(groups == g)[0] for g in unique}

    differences = []
    for _ in range(n_boot):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        index = np.concatenate([index_of[g] for g in drawn])
        mae_a = float(np.mean(np.abs(pred_a[index] - y_true[index])))
        mae_b = float(np.mean(np.abs(pred_b[index] - y_true[index])))
        differences.append(mae_a - mae_b)

    observed = (float(np.mean(np.abs(pred_a - y_true)))
                - float(np.mean(np.abs(pred_b - y_true))))
    low = float(np.percentile(differences, 2.5))
    high = float(np.percentile(differences, 97.5))
    return {
        "mae_difference": round(observed, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "favours_b": bool(low > 0),
    }


def compare_models(result, reference="pwer", challenger="ridge", n_boot=1000):
    """
    Test whether the challenger beats the baseline on the same segments.

    Args:
        result: Output of run_protocol.
        reference: Baseline model name.
        challenger: Model name to test.
        n_boot: Bootstrap replicates.

    Returns:
        Paired MAE difference with its interval.
    """
    meta = result["meta"]
    return paired_bootstrap(meta["y"], result["predictions"][reference],
                            result["predictions"][challenger],
                            meta["cv_group"], n_boot=n_boot)


def summarize(result, n_boot=500):
    """
    Build the comparison table of a protocol run.

    Args:
        result: Output of run_protocol.
        n_boot: Bootstrap replicates, 0 to skip.

    Returns:
        One row of metrics per model.
    """
    meta = result["meta"]
    y = meta["y"]
    table = {}

    for model, prediction in result["predictions"].items():
        metrics = regression_metrics(y, prediction)
        row = {
            "MAE": round(metrics["MAE"], 4),
            "RMSE": round(metrics["RMSE"], 4),
            "R2": round(metrics["R2"], 4),
            "bias": round(metrics["bias"], 4),
            "pearson": round(metrics["pearson"], 4),
            "spearman": round(metrics["spearman"], 4),
        }
        row.update(corpus_wer_gap(prediction, meta))
        if n_boot:
            interval = cluster_bootstrap(y, prediction, meta["cv_group"],
                                         n_boot=n_boot)
            row["MAE_ci"] = [interval["ci_low"], interval["ci_high"]]
        table[model] = row

    baseline = table.get("pwer", {}).get("MAE")
    for model, row in table.items():
        row["gain_vs_pwer"] = round(1.0 - row["MAE"] / baseline, 3) \
            if baseline else None

    return {
        "protocol": result["protocol"],
        "feature_set": result["feature_set"],
        "n_features": result["n_features"],
        "n_segments": int(len(y)),
        "wer_mean": round(float(y.mean()), 4),
        "wer_std": round(float(y.std()), 4),
        "models": table,
    }


def ridge_export(rows, blocks=("proxy", "text"), roles=PROXY_ROLES,
                 feature_set="transferable", alpha=None):
    """
    Fit the final ridge on the whole pool and export what internal code needs.

    The returned dictionary is plain JSON: feature order, scaler statistics,
    coefficients and intercept. No binary file to move.

    Args:
        rows: Labelled records.
        blocks: Feature blocks to include.
        roles: Proxy ASR systems.
        feature_set: "transferable" or "full".
        alpha: Fixed penalty, tuned by grouped search when None.

    Returns:
        Serializable model description.
    """
    X, names, meta = design(rows, blocks, roles, feature_set)
    y, groups = meta["y"], meta["cv_group"]

    pipeline = make_pipeline(StandardScaler(), Ridge())
    if alpha is None:
        search = GridSearchCV(pipeline, {"ridge__alpha": list(RIDGE_ALPHAS)},
                              cv=GroupKFold(3),
                              scoring="neg_mean_absolute_error")
        search.fit(X, y, groups=groups)
        pipeline = search.best_estimator_
        alpha = float(pipeline.named_steps["ridge"].alpha)
    else:
        pipeline.set_params(ridge__alpha=alpha).fit(X, y)

    scaler = pipeline.named_steps["standardscaler"]
    ridge = pipeline.named_steps["ridge"]

    return {
        "feature_names": names,
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in ridge.coef_],
        "intercept": float(ridge.intercept_),
        "alpha": float(alpha),
        "clip": list(WER_CLIP),
        "blocks": list(blocks),
        "roles": list(roles),
        "feature_set": feature_set,
        "n_train_segments": int(len(y)),
        "train_wer_mean": round(float(y.mean()), 4),
        "train_corpora": sorted(set(meta["corpus"].tolist())),
        "train_langs": sorted(set(meta["lang"].tolist())),
        "train_conditions": sorted(set(meta["condition"].tolist())),
    }


def apply_export(export, rows, blocks=None, roles=None):
    """
    Apply an exported ridge to new records, without scikit-learn.

    This is the function to reimplement internally: it is a dot product.

    Args:
        export: Output of ridge_export.
        rows: Records to score.
        blocks: Feature blocks, taken from the export when None.
        roles: Proxy ASR systems, taken from the export when None.

    Returns:
        Predicted WER per segment.
    """
    from features import build_features

    blocks = tuple(blocks or export["blocks"])
    roles = tuple(roles or export["roles"])
    X, _, _, _, names = build_features(rows, blocks=blocks, roles=roles)
    X = np.asarray(X, dtype=float)

    index = {name: i for i, name in enumerate(names)}
    missing = [n for n in export["feature_names"] if n not in index]
    if missing:
        raise ValueError(f"features missing from the records: {missing}")

    columns = X[:, [index[n] for n in export["feature_names"]]]
    standardized = (columns - np.array(export["scaler_mean"])) \
        / np.array(export["scaler_scale"])
    raw = standardized @ np.array(export["coef"]) + export["intercept"]
    return np.clip(raw, *export["clip"])


# --- transfer to internal data ------------------------------------------------


def predict_calls_export(export, rows):
    """
    Rank calls with an exported ridge, without joblib and without sklearn.

    Same output schema as deploy.predict_calls, so it is a drop-in replacement
    for pipeline.rank when the model travelled as coefficients rather than as a
    file.

    Args:
        export: Output of ridge_export.
        rows: Records to score.

    Returns:
        One entry per call, worst estimated WER first.
    """
    estimates = apply_export(export, rows)
    durations = np.array([r["duration"] for r in rows], dtype=float)
    groups = np.array([r["sample_id"] for r in rows])

    results = []
    for call in sorted(set(groups)):
        mask = groups == call
        weight = durations[mask].sum()
        if weight <= 0:
            continue
        results.append({
            "call": call,
            "n_segments": int(mask.sum()),
            "duration_s": round(float(weight), 1),
            "wer_estimated": round(
                float(np.sum(estimates[mask] * durations[mask]) / weight), 4),
        })
    results.sort(key=lambda item: -item["wer_estimated"])
    return results


def _aggregate_calls(y, prediction, rows):
    """
    Aggregate segment values into duration-weighted call values.

    Args:
        y: True segment WER.
        prediction: Predicted segment WER.
        rows: Records, providing duration and sample_id.

    Returns:
        True and predicted WER per call.
    """
    durations = np.array([r["duration"] for r in rows], dtype=float)
    groups = np.array([r["sample_id"] for r in rows])

    true_values, predicted_values = [], []
    for call in sorted(set(groups)):
        mask = groups == call
        weight = durations[mask].sum()
        if weight <= 0:
            continue
        true_values.append(float(np.sum(y[mask] * durations[mask]) / weight))
        predicted_values.append(
            float(np.sum(prediction[mask] * durations[mask]) / weight))
    return np.array(true_values), np.array(predicted_values)


def transfer_report(export, rows, roles=PROXY_ROLES, n_boot=500):
    """
    Score an exported ridge on labelled records it was not trained on.

    Reported next to the raw proxy disagreement, so that the transfer is judged
    against the baseline a practitioner would use without any model.

    calibration_slope is the ordinary least squares slope of the true WER
    regressed on the predicted one, the standard external-validation
    diagnostic: 1 means the dynamic range is preserved, below 1 means the
    predictions are too spread out, above 1 means they are compressed.

    Args:
        export: Output of ridge_export.
        rows: Labelled records from the target domain.
        roles: Proxy ASR systems.
        n_boot: Bootstrap replicates, 0 to skip.

    Returns:
        Segment-level and call-level metrics for the ridge and the baseline.
    """
    from features import proxy_features

    y = np.array([r["label_wer"] for r in rows], dtype=float)
    ridge_prediction = apply_export(export, rows)
    proxy_prediction = np.clip(
        np.array([proxy_features(r, roles).get("pwer_mean", 0.0) for r in rows],
                 dtype=float), *WER_CLIP)

    meta = {
        "n_hyp_words": np.array([max(r.get("n_hyp_words", 0), 1) for r in rows],
                                dtype=float),
        "n_ref_words": np.array([r["n_ref_words"] for r in rows], dtype=float),
        "label_errors": np.array([r["label_errors"] for r in rows], dtype=float),
    }
    groups = np.array([r["sample_id"] for r in rows])

    report = {
        "n_segments": len(rows),
        "n_calls": int(len(set(groups))),
        "wer_mean": round(float(y.mean()), 4),
        "wer_std": round(float(y.std()), 4),
        "trained_on": {
            "corpora": export.get("train_corpora"),
            "langs": export.get("train_langs"),
            "conditions": export.get("train_conditions"),
            "n_segments": export.get("n_train_segments"),
            "wer_mean": export.get("train_wer_mean"),
        },
        "segment": {},
        "call": {},
    }

    for name, prediction in (("ridge", ridge_prediction),
                             ("pwer", proxy_prediction)):
        metrics = regression_metrics(y, prediction)
        entry = {k: round(metrics[k], 4) for k in
                 ("MAE", "RMSE", "R2", "bias", "pearson", "spearman")}
        entry.update(corpus_wer_gap(prediction, meta))
        slope, intercept = np.polyfit(prediction, y, 1)
        entry["calibration_slope"] = round(float(slope), 3)
        entry["calibration_intercept"] = round(float(intercept), 4)
        if n_boot:
            interval = cluster_bootstrap(y, prediction, groups, n_boot=n_boot)
            entry["MAE_ci"] = [interval["ci_low"], interval["ci_high"]]
        report["segment"][name] = entry

        true_calls, predicted_calls = _aggregate_calls(y, prediction, rows)
        if true_calls.size >= 3:
            call_metrics_values = regression_metrics(true_calls, predicted_calls)
            report["call"][name] = {
                k: round(call_metrics_values[k], 4) for k in
                ("MAE", "RMSE", "R2", "bias", "pearson", "spearman")
            }

    return report


def fit_bundle(rows, model="ridge", blocks=("proxy", "text"), roles=PROXY_ROLES,
               feature_set="transferable", norm_config=None, seed=0, notes=""):
    """
    Train any regressor on a pool and return a bundle usable by deploy.

    This is the alternative to exporting ridge coefficients: when the
    transcription files themselves have been copied to the internal machine,
    the model can simply be retrained there, with no weights to transport and
    no restriction to linear models.

    The returned dictionary has the same shape as deploy.fit_final, so
    deploy.predict, deploy.predict_calls and pipeline.rank accept it as is.

    Args:
        rows: Labelled records.
        model: Model name from MODELS, excluding the mean and pwer baselines.
        blocks: Feature blocks to include.
        roles: Proxy ASR systems.
        feature_set: "transferable" or "full".
        norm_config: Normalization settings.
        seed: Random seed.
        notes: Free-form provenance note.

    Returns:
        Trained bundle.
    """
    X, names, meta = design(rows, blocks, roles, feature_set)
    y, groups = meta["y"], meta["cv_group"]
    estimator = _tuned_fit(model, X, y, np.arange(len(y)), groups, seed)

    return {
        "bundle_version": 1,
        "estimator": estimator,
        "feature_names": names,
        "blocks": tuple(blocks),
        "roles": tuple(roles),
        "target": "wer",
        "model": model,
        "feature_set": feature_set,
        "norm_config": dict(norm_config or {}),
        "n_train_segments": len(rows),
        "n_train_calls": len({r["sample_id"] for r in rows}),
        "train_wer_mean": float(np.mean(y)),
        "train_corpora": sorted(set(meta["corpus"].tolist())),
        "train_langs": sorted(set(meta["lang"].tolist())),
        "train_conditions": sorted(set(meta["condition"].tolist())),
        "notes": notes,
    }
