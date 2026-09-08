"""
Train, save, load, and deploy a WER prediction model for inference.
"""

import json
import os

import joblib
import numpy as np

from evaluate import make_model

# from features import PROXY_ROLES, block_of, proxy_features, text_features
from features import (PROXY_ROLES, block_of, extra_features,
                     proxy_features, text_features)

BUNDLE_VERSION = 1


# -----------------------------------------------------------------------------
# Feature building without labels
# -----------------------------------------------------------------------------

# def features_for_inference(rows, blocks=("proxy", "text"), roles=PROXY_ROLES):
#     """
#     Build inference features without reference labels.
#
#     Args:
#         rows: Input records.
#         blocks: Feature blocks to include.
#         roles: Proxy ASR systems.
#
#     Returns:
#         Feature dictionaries.
#     """
#     entries = []
#     for row in rows:
#         entry = {}
#         if "proxy" in blocks:
#             entry.update(proxy_features(row, roles))
#         if "text" in blocks:
#             entry.update(text_features(row))
#         entries.append(entry)
#     return entries


def features_for_inference(rows, blocks=("proxy", "text", "extra"),
                           roles=PROXY_ROLES):
    """
    Build inference features without reference labels.

    Args:
        rows: Input records.
        blocks: Feature blocks to include.
        roles: Proxy ASR systems.

    Returns:
        Feature dictionaries.
    """
    entries = []
    for row in rows:
        entry = {}
        if "proxy" in blocks:
            entry.update(proxy_features(row, roles))
        if "text" in blocks:
            entry.update(text_features(row))
        if "extra" in blocks:
            entry.update(extra_features(row, roles))
        entries.append(entry)
    return entries


def entries_to_matrix(entries, feature_names):
    """
    Convert feature dictionaries into a feature matrix.

    Args:
        entries: Feature dictionaries.
        feature_names: Ordered feature names.

    Returns:
        Feature matrix.
    """
    matrix = np.zeros((len(entries), len(feature_names)), dtype=np.float64)
    for i, entry in enumerate(entries):
        for j, name in enumerate(feature_names):
            matrix[i, j] = float(entry.get(name, 0.0))
    return matrix


# -----------------------------------------------------------------------------
# Training the final model
# -----------------------------------------------------------------------------


def fit_final(rows, blocks=("proxy", "text", "extra"), target="wer", model="hgb",  #blocks=("proxy", "text")
              roles=PROXY_ROLES, norm_config=None, seed=0, notes=""):
    """
    Train the final prediction model.

    Args:
        rows: Training records.
        blocks: Feature blocks to include.
        target: Prediction target.
        model: Regression model.
        roles: Proxy ASR systems.
        norm_config: Normalization settings.
        seed: Random seed.
        notes: Additional information.

    Returns:
        Trained model bundle.
    """
    entries = features_for_inference(rows, blocks, roles)
    feature_names = sorted({key for entry in entries for key in entry})
    feature_names = [n for n in feature_names if block_of(n) in blocks]
    matrix = entries_to_matrix(entries, feature_names)

    y = np.array(
        [r["label_errors"] if target == "errors" else r["label_wer"]
         for r in rows], dtype=float
    )

    estimator = make_model(model, seed)
    estimator.fit(matrix, y)

    return {
        "bundle_version": BUNDLE_VERSION,
        "estimator": estimator,
        "feature_names": feature_names,
        "blocks": tuple(blocks),
        "roles": tuple(roles),
        "target": target,
        "model": model,
        "norm_config": dict(norm_config or {}),
        "n_train_segments": len(rows),
        "n_train_calls": len({r["sample_id"] for r in rows}),
        "train_wer_mean": float(np.mean([r["label_wer"] for r in rows])),
        "notes": notes,
    }


def save_model(bundle, path):
    """
    Save a trained model bundle.

    Args:
        bundle: Model bundle.
        path: Output file path.

    Returns:
        Saved file path.
    """
    joblib.dump(bundle, path)
    sidecar = {k: v for k, v in bundle.items() if k != "estimator"}
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=1, ensure_ascii=False, default=str)
    return path


def load_model(path):
    """
    Load a saved model bundle.

    Args:
        path: Model file path.

    Returns:
        Loaded model bundle.
    """
    bundle = joblib.load(path)
    if bundle.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(
            f"bundle version {bundle.get('bundle_version')} != {BUNDLE_VERSION}"
        )
    return bundle


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


def predict(bundle, rows, norm_config=None):
    """
    Predict the WER for individual speech segments.

    Args:
        bundle: Trained model bundle.
        rows: Input records.
        norm_config: Normalization settings.

    Returns:
        Estimated WER for each segment.
    """
    if norm_config is not None and bundle["norm_config"] and \
            dict(norm_config) != bundle["norm_config"]:
        raise ValueError(
            f"normalization mismatch: model trained with "
            f"{bundle['norm_config']}, called with {dict(norm_config)}"
        )

    entries = features_for_inference(rows, bundle["blocks"], bundle["roles"])
    matrix = entries_to_matrix(entries, bundle["feature_names"])
    raw = bundle["estimator"].predict(matrix)

    if bundle["target"] == "errors":
        lengths = np.array([
            max(len((r.get("hyp_target_norm") or "").split()), 1) for r in rows
        ], dtype=float)
        return np.clip(np.clip(raw, 0, None) / lengths, 0.0, 2.0)
    return np.clip(raw, 0.0, 2.0)


def predict_calls(bundle, rows, norm_config=None):
    """
    Predict the WER at the call level; weighting segment estimates by their duration.

    Args:
        bundle: Trained model bundle.
        rows: Input records.
        norm_config: Normalization settings.

    Returns:
        Estimated WER for each call.
    """
    estimates = predict(bundle, rows, norm_config)
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
                float(np.sum(estimates[mask] * durations[mask]) / weight), 4
            ),
        })
    results.sort(key=lambda item: -item["wer_estimated"])
    return results