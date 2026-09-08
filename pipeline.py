"""
High-level pipeline for dataset preparation, transcription, feature extraction,
model evaluation, and deployment.

Usage
    records = pl.prepare(json_path, lang="es", out_dir=..., audio_root=...)
    pl.sanity(records)
    pl.transcribe(records, out_dir, ["target"])
    pl.accents(records, out_dir)
    pl.transcribe(records, out_dir)                    # proxies
    rows = pl.merge(records, out_dir)

    # analyse
    pl.ablate(rows, target="wer")
    pl.compare_proxies(rows)
    pl.evaluate(rows, target="wer")

    # deployement
    bundle = pl.fit(rows, out_dir=...)
    pl.rank(bundle, new_audio_to_predict)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from merge import build_table
from prepare_dataset import prepare_corpus, read_norm_config, read_records
from text_norm import diagnose_accent_impact
from transcribe import SYSTEMS, load_cache, run_all, sanity_check_canary


def report(title, pairs):
    """
    Display a formatted summary.

    Args:
        title: Report title.
        pairs: Key-value pairs to display.

    Returns:
        None.
    """
    print(f"\n{title}")
    width = max((len(str(k)) for k, _ in pairs), default=0)
    for key, value in pairs:
        print(f"  {str(key):<{width}} : {value}")


def load(out_dir):
    """
    Load the prepared dataset (the records written by prepare()).

    Args:
        out_dir: Output directory.

    Returns:
        Prepared records.
    """
    return read_records(os.path.join(out_dir, "segments.jsonl"))


def prepare(json_path, lang, out_dir, audio_root="", min_ref_words=10,
            roles=None, strip_accents=False, expand_numbers=True):  # try
    """
    Prepare the dataset for ASR transcription.

    Args:
        json_path: Dataset annotation file.
        lang: Language code.
        out_dir: Output directory.
        audio_root: Root audio directory.
        min_ref_words: Minimum reference length (after normalisation).
        roles: Speaker roles to include.
        strip_accents: Whether to remove diacritics.
        expand_numbers: Whether to expand numbers into words.

    Returns:
        Prepared records.
    """
    norm_config = {"strip_accents": strip_accents,
                   "expand_numbers": expand_numbers}

    records, stats, failures = prepare_corpus(
        json_path, lang=lang, out_dir=out_dir, audio_root=audio_root,
        min_ref_words=min_ref_words,
        roles=set(roles) if roles else None,
        norm_config=norm_config,
    )

    report("normalization (frozen in norm_config.json)",
           sorted(norm_config.items()))
    report("parsing", sorted(stats.items()))

    if failures:
        print(f"\n{len(failures)} export failures, first 5:")
        for item in failures[:5]:
            print(f"  {item}")

    if stats["too_short"] > stats["kept"]:
        print(f"\nWARNING: {stats['too_short']} segments dropped as too short "
              f"vs {stats['kept']} kept -> lower min_ref_words if your turns "
              f"are naturally brief.")

    print("\nfirst references, raw vs normalized:")
    for record in records[:3]:
        print(f"  raw  : {record['reference']}")
        print(f"  norm : {record['reference_norm']} "
              f"({record['n_ref_words']} words)")

    return records


def sanity(records, model="/domino/datasets/ModelHub-model-huggingface-nvidia/canary-180m-flash/main/canary-180m-flash.nemo", n_per_lang=20):
    """
    Run a sanity check on the ASR model.

    Args:
        records: Prepared records.
        model: ASR model.
        n_per_lang: Number of samples per language.

    Returns:
        Sample transcriptions.
    """
    cache = sanity_check_canary(records, base_model_path=model,
                                n_per_lang=n_per_lang)
    print("\nEvery hypothesis must be in the same language as its reference.")
    return cache


def transcribe(records, out_dir, systems=None):
    """
    Run ASR transcription.

    Args:
        records: Prepared records.
        out_dir: Output directory.
        systems: ASR systems to execute (list of roles among 'target', 'proxy_a', 'proxy_b').

    Returns:
        Transcription results.
    """
    selected = SYSTEMS if not systems else {r: SYSTEMS[r] for r in systems}
    results = run_all(records, os.path.join(out_dir, "asr"), systems=selected)
    report("transcribed", [(role, f"{len(texts)} segments")
                          for role, texts in results.items()])
    return results


def accents(records, out_dir, min_delta=0.005):
    """
    Evaluate the impact of accent normalization.

    Args:
        records: Prepared records.
        out_dir: Output directory.
        min_delta: Minimum WER improvement threshold.

    Returns:
        Accent normalization recommendations.
    """
    cache = load_cache(os.path.join(out_dir, "asr", "target.jsonl"))
    if not cache:
        print("No target transcriptions yet -> run transcribe(['target']).")
        return None

    by_lang = {}
    for record in records:
        row = cache.get(record["segment_id"])
        if row:
            by_lang.setdefault(record["lang"], []).append(
                (record["reference"], row["text"])
            )

    verdicts = {}
    for lang, pairs in by_lang.items():
        impact = diagnose_accent_impact(pairs, lang)
        delta = impact["keep"] - impact["strip"]
        verdicts[lang] = "strip_accents=True" if delta > min_delta else "keep as is"
        report(f"accent impact [{lang}] on {len(pairs)} pairs", [
            ("wer keeping accents", f"{impact['keep']:.4f}"),
            ("wer stripping", f"{impact['strip']:.4f}"),
            ("delta", f"{delta:.4f}"),
            ("verdict", verdicts[lang]),
        ])
    return verdicts


def merge(records, out_dir):
    """
    Merge transcriptions and generate WER labels.

    Args:
        records: Prepared records.
        out_dir: Output directory.

    Returns:
        Merged dataset.
    """
    asr_dir = os.path.join(out_dir, "asr")
    results = {}
    for role in SYSTEMS:
        cache = load_cache(os.path.join(asr_dir, f"{role}.jsonl"))
        if cache:
            results[role] = {sid: row["text"] for sid, row in cache.items()}

    if "target" not in results:
        print("No target transcriptions -> nothing to label.")
        return None

    rows, info = build_table(records, results, out_dir)
    report("normalization used", sorted(info["norm_config"].items()))
    report("coverage", [(role, f"{n}/{len(records)}")
                        for role, n in info["coverage"].items()])
    report("target WER on this corpus", sorted(info["summary"].items()))

    print("\nlabelled examples:")
    for row in rows[:2]:
        print(f"  ref    : {row['reference_norm']}")
        print(f"  target : {row['hyp_target_norm']}")
        print(f"  label  : {row['label_errors']} errors, "
              f"wer={row['label_wer']:.3f} "
              f"(S={row['label_S']} D={row['label_D']} I={row['label_I']})")

    print(f"\n{len(rows)} rows -> {os.path.join(out_dir, 'table.jsonl')}")
    return rows


def status(out_dir):
    """
    Summarize the current pipeline status.

    Args:
        out_dir: Output directory.

    Returns:
        Status information.
    """
    records = load(out_dir)
    lines = [
        ("segments", len(records)),
        ("calls", len({r["sample_id"] for r in records})),
        ("hours", round(sum(r["duration"] for r in records) / 3600.0, 3)),
        ("languages", sorted({r["lang"] for r in records})),
        ("norm_config", read_norm_config(out_dir)),
    ]

    for role in SYSTEMS:
        cache = load_cache(os.path.join(out_dir, "asr", f"{role}.jsonl"))
        lines.append((role, f"{len(cache)}/{len(records)}"))
    lines.append(("table built",
                  os.path.exists(os.path.join(out_dir, "table.jsonl"))))
    report(f"status of {out_dir}", lines)
    return dict(lines)


def to_frame(rows):
    """
    Convert records into a pandas DataFrame.

    Args:
        rows: Input records.

    Returns:
        DataFrame.
    """
    import pandas as pd

    return pd.DataFrame(rows)


def load_table(out_dir):
    """
    Load the merged analysis table (written by merge()).

    Args:
        out_dir: Output directory.

    Returns:
        Merged records.
    """
    from merge import read_table

    return read_table(out_dir)


def ablate(rows, target="wer", n_splits=5):
    """
    Run feature ablation experiments.

    Args:
        rows: Input records.
        target: Prediction target.
        n_splits: Number of cross-validation folds.

    Returns:
        Ablation results.
    """
    import pandas as pd

    from evaluate import ablation, denominator_bias

    report("denominator drift (n_ref / n_hyp)",
           sorted(denominator_bias(rows).items()))
    table = pd.DataFrame(ablation(rows, target=target, n_splits=n_splits))
    print(f"\nablation, target={target}, grouped by call")
    print(table.to_string(index=False))
    return table


def evaluate(rows, target="wer", model="hgb", blocks=("proxy", "text", "extra"),
             n_splits=5, importance=True, csv_dir=None):
    """
    Evaluate a WER prediction model.

    Args:
        rows: Input records.
        target: Prediction target.
        model: Regression model.
        blocks: Feature blocks to include.
        n_splits: Number of cross-validation folds.
        importance: Whether to compute feature importance.

    Returns:
        Evaluation results.
    """
    import pandas as pd

    from evaluate import (call_metrics, cross_validate, permutation_importance,
                          segment_metrics)

    result = cross_validate(rows, blocks=blocks, target=target, model=model,
                            n_splits=n_splits)
    report(f"segment level ({result['n_folds']} folds, "
           f"{result['n_features']} features)",
           [(k, round(v, 4)) for k, v in segment_metrics(result).items()])

    calls, summary = call_metrics(result)
    print("\nper call, duration-weighted")
    print(pd.DataFrame(calls).to_string(index=False))
    report("call level", [(k, round(v, 4) if isinstance(v, float) else v)
                          for k, v in summary.items()])

    if csv_dir is not None:
        import os

        os.makedirs(csv_dir, exist_ok=True)
        path = os.path.join(csv_dir, f"calls_{target}_{model}.csv")
        pd.DataFrame(calls).to_csv(path, index=False)
        print(f"\nsaved to {path}")

    if importance:
        base, scores = permutation_importance(
            rows, blocks=blocks, target=target, model=model, n_splits=n_splits, n_repeats=3
        )
        report(f"permutation importance (base MAE {base:.4f})",
               [(name, f"+{delta:.4f}") for name, delta, _ in scores[:10]])

    return result


def compare_proxies(rows, target="wer", model="hgb", n_splits=5):
    """
    Compare different proxy ASR configurations.

    Args:
        rows: Input records.
        target: Prediction target.
        model: Regression model.
        n_splits: Number of cross-validation folds.

    Returns:
        Proxy comparison results.
    """
    import pandas as pd

    from evaluate import single_proxy_ablation

    table = pd.DataFrame(single_proxy_ablation(rows, target=target, model=model,
                                               n_splits=n_splits))
    print("\nproxy contribution")
    print(table.to_string(index=False))
    return table


# -----------------------------------------------------------------------------
# Deployment
# -----------------------------------------------------------------------------


def fit(rows, out_dir=None, target="wer", model="hgb", blocks=("proxy", "text", "extra"),
        notes=""):
    """
    Train the final WER prediction model.

    Args:
        rows: Training records.
        out_dir: Output directory.
        target: Prediction target.
        model: Regression model.
        blocks: Feature blocks to include.
        notes: Additional information.

    Returns:
        Trained model bundle.
    """
    from prepare_dataset import read_norm_config
    from deploy import fit_final, save_model

    norm_config = read_norm_config(out_dir) if out_dir else None
    bundle = fit_final(rows, blocks=blocks, target=target, model=model,
                       norm_config=norm_config, notes=notes)
    report("fitted model", [
        ("segments", bundle["n_train_segments"]),
        ("calls", bundle["n_train_calls"]),
        ("features", len(bundle["feature_names"])),
        ("target", bundle["target"]),
        ("model", bundle["model"]),
        ("norm_config", bundle["norm_config"]),
    ])
    if out_dir:
        path = save_model(bundle, os.path.join(out_dir, "wer_model.joblib"))
        print(f"\nsaved to {path} (+ .json sidecar for auditing)")
    return bundle


def rank(bundle, rows):
    """
    Rank calls by their estimated WER.

    Args:
        bundle: Trained model bundle.
        rows: Input records.

    Returns:
        Ranked calls.

    Note: rows only require hyp_target_norm, the proxy hypotheses, duration and
    sample_id.
    """
    import pandas as pd
    from deploy import predict_calls

    table = pd.DataFrame(predict_calls(bundle, rows))
    print(table.to_string(index=False))
    return table


def report_metrics(rows, target="wer", model="hgb", blocks=("proxy", "text", "extra"),
                   n_splits=5, threshold=0.25, ks=(50, 100, 200), weighting="duration"):
    import numpy as np
    import pandas as pd

    from evaluate import (call_metrics, cross_validate, full_report,
                          precision_at_k, r2_decomposition) #stratification_check

    result = cross_validate(rows, blocks=blocks, target=target, model=model,
                            n_splits=n_splits)
    calls, _ = call_metrics(result)

    levels = {
        "segment": (result["y_wer"], result["pred_wer"]),
        "call": (np.array([c["wer_true"] for c in calls]),
                 np.array([c["wer_pred"] for c in calls])),
    }
    reports = {level: full_report(y_true, y_pred, threshold=threshold)
               for level, (y_true, y_pred) in levels.items()}

    print("\nregression")
    print(pd.DataFrame({k: reports[k]["regression"] for k in levels}).T
          .round(4).to_string())

    print(f"\ngood / bad transcriptions at WER > {threshold}")
    print(pd.DataFrame({k: reports[k]["binary"] for k in levels}).T
          .round(4).to_string())

    for level in levels:
        print(f"\nconfusion matrix - {level} (rows = true, cols = predicted)")
        print(pd.DataFrame(reports[level]["confusion_matrix"],
                           index=["good", "bad"],
                           columns=["good", "bad"]).to_string())

    y_true, y_pred = levels["segment"]
    print("\nselection quality (segments, ranked worst first)")
    print(pd.DataFrame([precision_at_k(y_true, y_pred, k, threshold)
                        for k in ks]).round(4).to_string(index=False))

    # Note: lignes 499-512 non visibles entre captures 4835 et 4836 (probablement affichage r2_decomposition)
    return reports


# -----------------------------------------------------------------------------
# Feature importance
# -----------------------------------------------------------------------------

BNP_GREEN, BNP_DARK, GREY, CAPTION = "#00915A", "#006748", "#B0B7BC", "#5A6468"


def plot_ridge(table, n=12, path=None):
    """
    Plot Ridge regression coefficients.

    Args:
        table: Ridge coefficients.
        n: Number of features to display.
        path: Output file path.

    Returns:
        Matplotlib figure.
    """
    df = pd.DataFrame(table).head(n).iloc[::-1]        # reversed: largest on top
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(df) + 1.4))
    colors = [BNP_GREEN if c > 0 else GREY for c in df["coef"]]
    ax.barh(df["feature"], df["coef"], xerr=df["std"], color=colors,
            error_kw={"ecolor": BNP_DARK, "capsize": 3, "lw": 1})
    ax.axvline(0, color=BNP_DARK, lw=0.8)
    ax.set_xlabel("Ridge coefficient") # on standardised features
    ax.set_title("Linear contribution of each feature to the estimated WER", loc="left", fontweight="bold", pad=12)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_pdp(curve, feature, path=None):
    """
    Plot a partial dependence curve for one feature.

    Args:
        curve: Partial dependence values.
        feature: Feature name.
        path: Output file path.

    Returns:
        Matplotlib figure.
    """
    df = pd.DataFrame(curve)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(df["value"], df["predicted"], color=BNP_GREEN, lw=2.2,
            marker="o", ms=4.5, mfc="white", mec=BNP_GREEN, mew=1.6)
    ax.set_xlabel(feature.replace("_", " "))
    ax.set_ylabel(f"Estimated WER by {feature.replace('_', ' ')}")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_importance_comparison(permutation_scores, xgb_gains, n=10, path=None):
    """
    Compare permutation and XGBoost feature importance.

    Args:
        permutation_scores: Permutation importance scores.
        xgb_gains: XGBoost importance scores.
        n: Number of features to display.
        path: Output file path.

    Returns:
        Matplotlib figure.

    Note : Divergence is expected; permutation discounts correlated features, gain
    """
    perm = {name: delta for name, delta, _ in permutation_scores}
    gain = {row["feature"]: row["gain"] for row in xgb_gains}
    order = sorted(perm, key=lambda k: -perm[k])[:n][::-1]

    perm_max = max(perm.values()) or 1
    gain_max = max(gain.values()) or 1
    y = range(len(order))

    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(order) + 1.6))
    ax.barh([i + 0.2 for i in y], [perm[f] / perm_max for f in order],
            height=0.38, color=BNP_GREEN, label="Permutation (out-of-fold)")
    ax.barh([i - 0.2 for i in y], [gain.get(f, 0) / gain_max for f in order],
            height=0.38, color=GREY, label="XGBoost gain (in-sample)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(order)
    ax.set_xlabel("Relative importance (normalised to the maximum)")
    ax.set_title("Feature importance",
                 loc="left", fontweight="bold", pad=12)

    ax.legend(frameon=False, loc="lower right", fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_distributions(rows, target="wer", model="hgb",
                       blocks=("proxy", "text", "extra"), n_splits=5, clip=1.5,
                       path=None):
    """True vs estimated WER distributions, at segment and call level. """
    from evaluate import call_metrics, cross_validate
    from plots_dist import plot_wer_distributions

    result = cross_validate(rows, blocks=blocks, target=target, model=model,
                            n_splits=n_splits)
    calls, _ = call_metrics(result)
    return plot_wer_distributions(result, calls, clip=clip, path=path)