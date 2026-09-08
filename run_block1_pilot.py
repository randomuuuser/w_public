"""
Block 1 pilot: validate the public-data plumbing end to end.

Scope on purpose: Spanish, FLEURS + VoxPopuli test splits, 300 segments each,
three ASR systems, no degradation. The goal is a working pipeline and a WER
sanity check, not a result.

One step per model, because the three systems do not share a dependency set:

    step "parakeet" -> nemo-toolkit==2.1.0, torch==2.7.0
    step "canary"   -> nemo_toolkit[asr]>=2.6.0
    step "whisper"  -> transformers>=4.46

Install the requirements of one step, restart the runtime, run the step. A
runtime restart keeps the files under /content, so the WAV scratch survives and
prepare() does not need to run again.

Each step appends to its own JSONL cache and skips segments already
transcribed, so an interrupted session is resumed by rerunning the same call.
"""

import json
import os

import numpy as np

import transcribe
from merge import build_table, corpus_summary
from public_data import build_public_corpus, has_sessions
from prepare_dataset import read_records
from text_norm import wer_counts

# --- configuration -----------------------------------------------------------

DRIVE_ROOT = "/content/drive/MyDrive/Travail/WER_Predictor/wer_public"
SCRATCH_ROOT = "/content/wav_cache"

PILOT = [("fleurs", "es", 300), ("voxpopuli", "es", 300)]

# Role names must match features.PROXY_ROLES and merge.TARGET_ROLE.
# "kind" selects both the runner and the step, so permuting roles never
# desynchronizes anything: the step is derived, never hardcoded.
SYSTEMS = {
    "target":  ("nvidia/parakeet-tdt-0.6b-v3", "parakeet"), 
    "proxy_a": ("nvidia/canary-1b-v2", "canary"),
    "proxy_b": ("openai/whisper-large-v3", "whisper"),
}

RUNNERS = {
    "canary": "run_canary",
    "parakeet": "run_parakeet",
    "whisper": "run_whisper",
}


def corpus_dirs(pilot=PILOT, drive_root=DRIVE_ROOT):
    """
    Resolve the output directory of each pilot corpus without touching audio.

    Args:
        pilot: List of (corpus, lang, n_segments).
        drive_root: Persistent root directory.

    Returns:
        Output directory per corpus tag.
    """
    return {f"{corpus}__{lang}__clean":
            os.path.join(drive_root, f"{corpus}__{lang}__clean")
            for corpus, lang, _ in pilot}


def roles_for(kind, systems=None):
    """
    List the roles served by a given model kind.

    Args:
        kind: Model kind, one of RUNNERS.
        systems: Role to (model id, kind) mapping.

    Returns:
        Roles to run in that step.
    """
    systems = systems or SYSTEMS
    if kind not in RUNNERS:
        raise ValueError(f"unknown kind {kind}, expected one of {list(RUNNERS)}")
    return tuple(role for role, (_, k) in systems.items() if k == kind)


# --- preparation -------------------------------------------------------------


def prepare(pilot=PILOT, drive_root=DRIVE_ROOT, scratch_root=SCRATCH_ROOT):
    """
    Stream the pilot corpora and write their records.

    Args:
        pilot: List of (corpus, lang, n_segments).
        drive_root: Persistent root directory.
        scratch_root: Local root directory for WAV files.

    Returns:
        Output directory per corpus tag.
    """
    dirs = {}
    for corpus, lang, n in pilot:
        _, out_dir, stats = build_public_corpus(
            corpus, lang, n, drive_root, scratch_root, min_ref_words=0,
        )
        dirs[f"{corpus}__{lang}__clean"] = out_dir
        print(f"\n=== {corpus} / {lang} ===")
        print(json.dumps(stats, indent=1))
    return dirs


# --- transcription -----------------------------------------------------------


def run_step(kind, dirs=None, systems=None, batch_size=8, chunk=128):
    """
    Run every role served by one model kind, over every pilot corpus.

    Args:
        kind: Model kind, one of RUNNERS.
        dirs: Output directory per corpus tag.
        systems: Role to (model id, kind) mapping.
        batch_size: Batch size.
        chunk: Number of records between cache flushes.

    Returns:
        Transcriptions per corpus tag and role.
    """
    systems = systems or SYSTEMS
    dirs = dirs or corpus_dirs()
    roles = roles_for(kind, systems)
    if not roles:
        print(f"no role uses kind {kind}, nothing to do")
        return {}

    runner = getattr(transcribe, RUNNERS[kind])
    results = {}

    for tag, out_dir in dirs.items():
        records = read_records(os.path.join(out_dir, "records.jsonl"))
        missing = [r for r in records if not os.path.exists(r["wav_path"])]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} WAV files missing under {SCRATCH_ROOT}, "
                f"rerun prepare() first"
            )
        for role in roles:
            model_id = systems[role][0]
            cache_path = os.path.join(out_dir, f"{role}.jsonl")
            print(f"\n=== {tag} | {role}: {model_id} | {len(records)} segments ===")
            cache = runner(records, cache_path, base_model_path=model_id,
                           batch_size=batch_size, chunk=chunk)
            results.setdefault(tag, {})[role] = {
                sid: row["text"] for sid, row in cache.items()
            }

    return results


def load_results(out_dir, roles):
    """
    Load every transcription cache written so far.

    Args:
        out_dir: Corpus directory.
        roles: Roles to load.

    Returns:
        Transcriptions per role.
    """
    results = {}
    for role in roles:
        cache = transcribe.load_cache(os.path.join(out_dir, f"{role}.jsonl"))
        if cache:
            results[role] = {sid: row["text"] for sid, row in cache.items()}
    return results


# --- sanity checks -----------------------------------------------------------


def wer_by_system(rows, roles):
    """
    Compute the word-weighted WER of every system against the reference.

    Args:
        rows: Merged records holding hyp_<role>_norm fields.
        roles: Roles to score.

    Returns:
        WER and coverage per role.
    """
    report_rows = {}
    for role in roles:
        errors, ref_words, covered = 0, 0, 0
        for row in rows:
            hypothesis = row.get(f"hyp_{role}_norm")
            if hypothesis is None:
                continue
            counts = wer_counts(row["reference_norm"], hypothesis)
            errors += counts["errors"]
            ref_words += counts["n_ref"]
            covered += 1
        report_rows[role] = {
            "wer": round(errors / ref_words, 4) if ref_words else None,
            "coverage": covered,
        }
    return report_rows


def label_spread(rows, sessions=True):
    """
    Report the WER dispersion of the target system.

    Variance is decomposed because it is additive and the standard deviation is
    not, following the law of total variance: Var(Y) = Var(E[Y|session]) +
    E[Var(Y|session)]. The ratio of the first term to the total is the
    intraclass correlation coefficient (Shrout and Fleiss, 1979), that is the
    share of WER variability occurring between sessions rather than within one.

    Args:
        rows: Labelled records.
        sessions: Whether the corpus exposes real recording sessions.

    Returns:
        Segment-level dispersion, and the session-level decomposition when
        sessions are available.
    """
    segment = np.array([r["label_wer"] for r in rows], dtype=float)
    spread = {
        "segment_wer_mean": round(float(segment.mean()), 4),
        "segment_wer_std": round(float(segment.std()), 4),
        "segment_wer_var": round(float(segment.var()), 6),
    }
    if not sessions:
        spread["sessions"] = None
        return spread

    by_session = {}
    for row in rows:
        by_session.setdefault(row["sample_id"], []).append(row["label_wer"])

    means = np.array([np.mean(v) for v in by_session.values()], dtype=float)
    sizes = np.array([len(v) for v in by_session.values()], dtype=float)
    within = np.array([np.var(v) for v in by_session.values()], dtype=float)

    var_between = float(np.average((means - segment.mean()) ** 2, weights=sizes))
    var_within = float(np.average(within, weights=sizes))
    var_total = var_between + var_within

    spread["sessions"] = {
        "n_sessions": int(means.size),
        "segments_per_session_mean": round(float(sizes.mean()), 1),
        "session_wer_mean": round(float(means.mean()), 4),
        "session_wer_std": round(float(means.std()), 4),
        "var_between": round(var_between, 6),
        "var_within": round(var_within, 6),
        "icc": round(var_between / var_total, 4) if var_total else None,
    }
    return spread


def show_samples(rows, roles, n=5):
    """
    Print reference and hypotheses side by side for a few segments.

    Args:
        rows: Merged records.
        roles: Roles to display.
        n: Number of segments to print.

    Returns:
        None.
    """
    for row in rows[:n]:
        print(f"\n[{row['lang']}] {row['segment_id']}  ({row['duration']}s)")
        print(f"  REF        : {row['reference_norm']}")
        for role in roles:
            print(f"  {role:<10} : {row.get(f'hyp_{role}_norm')}")


def report(out_dir, systems=None, target_role="target"):
    """
    Merge the transcriptions and print the block 1 exit criteria.

    Args:
        out_dir: Corpus directory.
        systems: Role to (model id, kind) mapping.
        target_role: Role scored as the target system.

    Returns:
        Merged rows and the report.
    """
    systems = systems or SYSTEMS
    records = read_records(os.path.join(out_dir, "records.jsonl"))
    results = load_results(out_dir, tuple(systems))
    if target_role not in results:
        raise ValueError(f"{target_role} has no transcription cache in {out_dir}")

    rows, meta = build_table(records, results, out_dir, target_role=target_role)
    corpus = records[0]["corpus"]

    summary = {
        "corpus": corpus,
        "coverage": meta["coverage"],
        "wer_by_system": wer_by_system(rows, list(results)),
        "target_summary": corpus_summary(rows, target_role),
        "spread": label_spread(rows, sessions=has_sessions(corpus)),
    }
    print(f"\n=== {os.path.basename(out_dir)} ===")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return rows, summary


# --- usage -------------------------------------------------------------------
#
# Common preamble, every session.
#
#   import sys
#   from google.colab import drive; drive.mount('/content/drive')
#   project_path = '/content/drive/MyDrive/Travail/WER_Predictor'
#   if project_path not in sys.path:
#       sys.path.append(project_path)
#
# Step 0, preparation. No GPU, no ASR dependency, ~2 min.
#
#   !pip install -q "datasets<4.0.0" soundfile librosa jiwer num2words
#   import run_block1_pilot as pilot
#   dirs = pilot.prepare()
#
# Step 1, parakeet. Install, then Runtime > Restart session, then run.
#
#   !pip install -q -r /content/drive/MyDrive/Travail/WER_Predictor/requirements_parakeet.txt
#   import run_block1_pilot as pilot
#   pilot.run_step("parakeet")
#
# Step 2, canary. Same cycle: install, restart, run.
#
#   !pip install -q -r /content/drive/MyDrive/Travail/WER_Predictor/requirements_canary.txt
#   import run_block1_pilot as pilot
#   pilot.run_step("canary")
#
# Step 3, whisper. Same cycle.
#
#   !pip install -q -r /content/drive/MyDrive/Travail/WER_Predictor/requirements_whisper.txt
#   import run_block1_pilot as pilot
#   pilot.run_step("whisper", batch_size=4)
#
# Report. No GPU needed, runs after any step.
#
#   import run_block1_pilot as pilot
#   for out_dir in pilot.corpus_dirs().values():
#       rows, summary = pilot.report(out_dir)
#       pilot.show_samples(rows, list(pilot.SYSTEMS), n=3)
#
# Smoke test before spending GPU time: 30 segments, one corpus, one model.
#
#   dirs = pilot.prepare(pilot=[("fleurs", "es", 30)])
#   pilot.run_step("parakeet", dirs=dirs)
#   pilot.report(dirs["fleurs__es__clean"], target_role="proxy_a")
#
# If the runtime is disconnected (not merely restarted), /content is wiped:
# rerun prepare(), the segment ids are deterministic and the caches on Drive
# stay valid.
