"""
Merge normalized transcriptions and generate the final dataset used for
feature extraction and model training.
"""

import json
import os

from prepare_dataset import read_norm_config
from text_norm import normalize_text, wer_counts

TARGET_ROLE = "target"


def merge_transcriptions(records, results, norm_config):
    """
    Merge ASR hypotheses with the reference records.

    Args:
        records: Reference records.
        results: Transcription results by system. format : {role: {segment_id: raw_text}}
        norm_config: Text normalization settings.

    Returns:
        Tuple containing merged records and transcription coverage.
    """
    coverage = {role: 0 for role in results}
    rows = []

    for record in records:
        row = dict(record)
        for role, by_segment in results.items():
            raw = by_segment.get(record["segment_id"])
            if raw is None:
                row[f"hyp_{role}"] = None
                row[f"hyp_{role}_norm"] = None
                continue
            coverage[role] += 1
            row[f"hyp_{role}"] = raw
            row[f"hyp_{role}_norm"] = normalize_text(
                raw, record["lang"], **norm_config
            )
        rows.append(row)

    return rows, coverage


def add_labels(rows, target_role=TARGET_ROLE):
    """
    Compute WER labels for the target ASR system.

    Args:
        rows: Merged records.
        target_role: Target ASR system.

    Returns:
        Labeled records.
    """
    labelled = []
    for row in rows:
        hypothesis = row.get(f"hyp_{target_role}_norm")
        if hypothesis is None:
            continue
        counts = wer_counts(row["reference_norm"], hypothesis)
        row.update({
            "label_errors": counts["errors"],
            "label_wer": counts["wer"],
            "label_S": counts["S"],
            "label_D": counts["D"],
            "label_I": counts["I"],
            "n_hyp_words": counts["n_hyp"],
        })
        labelled.append(row)
    return labelled


def corpus_summary(rows, target_role=TARGET_ROLE):
    """
    Compute corpus-level WER statistics of the target system, segment- and duration-weighted.

    Args:
        rows: Labeled records.
        target_role: Target ASR system.

    Returns:
        Summary statistics for the corpus.
    """
    if not rows:
        return {}

    total_errors = sum(r["label_errors"] for r in rows)
    total_ref = sum(r["n_ref_words"] for r in rows)
    total_duration = sum(r["duration"] for r in rows)

    return {
        "n_segments": len(rows),
        "hours": round(total_duration / 3600.0, 3),
        "wer_word_weighted": round(total_errors / total_ref, 4) if total_ref else None,
        "wer_segment_mean": round(sum(r["label_wer"] for r in rows) / len(rows), 4),
        "wer_duration_weighted": round(
            sum(r["label_wer"] * r["duration"] for r in rows) / total_duration, 4
        ) if total_duration else None,
        "pct_zero_wer": round(
            sum(1 for r in rows if r["label_errors"] == 0) / len(rows), 4
        ),
    }


def build_table(records, results, out_dir, target_role=TARGET_ROLE):
    """
    Build and save the final analysis table.

    Args:
        records: Reference records.
        results: Transcription results.
        out_dir: Output directory.
        target_role: Target ASR system.

    Returns:
        Analysis table and processing metadata.
    """
    norm_config = read_norm_config(out_dir)
    rows, coverage = merge_transcriptions(records, results, norm_config)
    rows = add_labels(rows, target_role=target_role)

    path = os.path.join(out_dir, "table.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return rows, {
        "coverage": coverage,
        "norm_config": norm_config,
        "summary": corpus_summary(rows, target_role),
    }


def read_table(out_dir):
    """
    Load the analysis table.

    Args:
        out_dir: Output directory.

    Returns:
        List of table records.
    """
    path = os.path.join(out_dir, "table.jsonl")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]