"""
Extract feature vectors from normalized ASR hypotheses for WER prediction.
"""

import gzip
import math
from collections import Counter

from text_norm import wer_counts

PROXY_ROLES = ("proxy_a", "proxy_b")


def _char_error_rate(reference, hypothesis):
    """
    Compute the Character Error Rate (CER) between two normalized transcripts.

    Args:
        reference: Reference transcript.
        hypothesis: Hypothesis transcript.

    Returns:
        Character Error Rate.
    """
    ref_chars = " ".join(reference.replace(" ", ""))
    hyp_chars = " ".join(hypothesis.replace(" ", ""))
    counts = wer_counts(ref_chars, hyp_chars)
    return counts["wer"]


def proxy_features(row, roles=PROXY_ROLES):
    """
    Compute agreement-based features between the target and proxy hypotheses.

    Args:
        row: Input record.
        roles: Proxy ASR systems.

    Returns:
        Dictionary of proxy-based features.
    """
    target = row.get("hyp_target_norm") or ""
    features = {}
    per_role = {}

    for role in roles:
        proxy = row.get(f"hyp_{role}_norm")
        if proxy is None:
            continue
        counts = wer_counts(proxy, target)
        n_ref = max(counts["n_ref"], 1)
        per_role[role] = counts["wer"]
        features.update({
            f"pwer_{role}": counts["wer"],
            f"pcer_{role}": _char_error_rate(proxy, target),
            f"psub_{role}": counts["S"] / n_ref,
            f"pdel_{role}": counts["D"] / n_ref,
            f"pins_{role}": counts["I"] / n_ref,
            f"plen_ratio_{role}": counts["n_hyp"] / n_ref,
        })

    if per_role:
        values = list(per_role.values())
        features["pwer_mean"] = sum(values) / len(values)
        features["pwer_min"] = min(values)
        features["pwer_max"] = max(values)
        features["pwer_spread"] = max(values) - min(values)

    # Proxy-vs-proxy: intrinsic segment difficulty, target-independent
    if len(roles) >= 2:
        first = row.get(f"hyp_{roles[0]}_norm")
        second = row.get(f"hyp_{roles[1]}_norm")
        if first is not None and second is not None:
            features["pwer_proxy_vs_proxy"] = wer_counts(first, second)["wer"]

    return features


def text_features(row):
    """
    Compute text-based features from the target hypothesis.

    Args:
        row: Input record.

    Returns:
        Dictionary of text-based features.
    """
    text = row.get("hyp_target_norm") or ""
    words = text.split()
    n_words = len(words)
    n_chars = len(text)
    duration = max(float(row.get("duration") or 0.0), 1e-6)

    encoded = text.encode("utf-8")
    compressed = len(gzip.compress(encoded)) if encoded else 0
    gzip_ratio = compressed / len(encoded) if encoded else 0.0

    counts = Counter(words)
    repeated = sum(c for c in counts.values() if c > 1)

    return {
        "n_hyp_words": n_words,
        "n_hyp_chars": n_chars,
        "duration": duration,
        "words_per_second": n_words / duration,
        "chars_per_word": n_chars / n_words if n_words else 0.0,
        "gzip_ratio": gzip_ratio,
        "type_token_ratio": len(counts) / n_words if n_words else 0.0,
        "repeated_word_ratio": repeated / n_words if n_words else 0.0,
        "max_word_repeat": max(counts.values()) if counts else 0,
        "log_duration": math.log1p(duration),
    }


# FEATURE_BLOCKS = {

# -----------------------------------------------------------------------------
# Add extra features
# -----------------------------------------------------------------------------


def _repeated_ngram_ratio(words, n):
    """Share of n-grams in the hypothesis that occur more than once."""
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(grams)


def extra_features(row, roles=PROXY_ROLES):
    """
    Compute additional features.

    Args:
        row: Input record containing hypothesis and role data.
        roles: List of proxy ASR system roles to compare against.

    Returns:
        Dictionary of extra features including:
        - x_is_operator: 1.0 if the speaker is an operator, else 0.0.
    """
    hypothesis = row.get("hyp_target_norm") or ""
    words = hypothesis.split()

    role = str(row.get("speaker_role") or "").strip().lower()
    features = {
        "x_is_operator": float(role.startswith("oper")),
        # "x_repeated_bigram_ratio": _repeated_ngram_ratio(words, 2),
        # "x_repeated_trigram_ratio": _repeated_ngram_ratio(words, 3),
    }

    proxy_vocabulary = set()
    for proxy_role in roles:
        proxy_text = row.get(f"hyp_{proxy_role}_norm")
        if proxy_text:
            proxy_vocabulary.update(proxy_text.split())

    # if words and proxy_vocabulary:
    #     orphans = sum(1 for word in words if word not in proxy_vocabulary)
    #     features["x_orphan_word_ratio"] = orphans / len(words)
    #     features["x_orphan_word_count"] = float(orphans)
    # else:
    #     features["x_orphan_word_ratio"] = 0.0
    #     features["x_orphan_word_count"] = 0.0

    # written by add_call_context, absent if it was not run
    # for key in ("ctx_pwer_prev", "ctx_pwer_next", "ctx_pwer_call",
    #             "ctx_position"):
    #     features[f"x_{key}"] = float(row.get(key, 0.0))

    return features


# def add_call_context(rows, roles=PROXY_ROLES):
#     """
#     Inject neighboring-segment context features into each row in place.
#
#     Uses temporal neighbors and call-level averages of proxy system disagreement
#     (pWER) to provide prior difficulty signals.
#
#     Args:
#         rows: List of input records.
#         roles: List of proxy ASR system roles.


FEATURE_BLOCKS = {
    "proxy": proxy_features,
    "text": text_features,
    "extra": extra_features,
}


def build_features(rows, blocks=("proxy", "text", "extra"), roles=PROXY_ROLES):
    """
    Build the feature matrix and target vectors.

    Args:
        rows: Input records.
        blocks: Feature blocks to include.
        roles: Proxy ASR systems.

    Returns:
        Feature matrix, target vectors, groups, and feature names.
    """
    matrix, y_errors, y_wer, groups = [], [], [], []

    for row in rows:
        features = {}
        for name in blocks:
            builder = FEATURE_BLOCKS[name]
            features.update(
                builder(row, roles) if name in ("proxy", "extra")
                else builder(row)
            )
        matrix.append(features)
        y_errors.append(float(row["label_errors"]))
        y_wer.append(float(row["label_wer"]))
        groups.append(row["sample_id"])

    names = sorted({key for features in matrix for key in features})
    X = [[features.get(name, 0.0) for name in names] for features in matrix]
    return X, y_errors, y_wer, groups, names


def block_of(name):
    """Which block a feature name belongs to, for the ablation."""
    if name.startswith("x_"):
        return "extra"
    return "proxy" if name.startswith(("pwer", "pcer", "psub", "pdel",
                                       "pins", "plen")) else "text"