"""
Driver for the public evaluation, blocks 2 and 3. Supersedes run_block1_pilot.

Block 2, controlled variance:
    same corpora as the pilot, several acoustic conditions, one directory per
    condition. Answers: does degrading the audio produce a WER range wide
    enough to train and evaluate on.

Block 3, generalization:
    more languages and corpora, then four protocols (group, lodo, lolo, loco)
    with the baselines. Answers: does the estimator survive a change of corpus,
    of language, of acoustic condition.

One step per ASR model, as in block 1, because the three systems do not share a
dependency set. Install the requirements of one step, restart the runtime, run
the step. A runtime restart keeps /content, so the WAV scratch survives.

Everything is resumable: records and hypothesis caches live on Drive, keyed by
segment id, and both prepare() and run_step() skip what already exists.
"""

import json
import os

import evaluate_public as ep
import transcribe
from merge import build_table
from prepare_dataset import read_records
from public_data import build_public_corpus

# --- configuration -----------------------------------------------------------

DRIVE_ROOT = "/content/drive/MyDrive/Travail/WER_Predictor/wer_public"

# WAV files live on Drive, not on /content: streaming and degrading is the slow
# part, and a disconnected Colab session wipes the local disk. Roughly 350 kB
# per 11 s segment, so about 70 MB per 200-segment configuration. Use
# audio_usage() before scaling a plan up.
SCRATCH_ROOT = os.path.join(DRIVE_ROOT, "wav")

SYSTEMS = {
    "target": ("nvidia/parakeet-tdt-0.6b-v3", "parakeet"),
    "proxy_a": ("nvidia/canary-1b-v2", "canary"),
    "proxy_b": ("openai/whisper-large-v3", "whisper"),
}

RUNNERS = {"canary": "run_canary", "parakeet": "run_parakeet",
           "whisper": "run_whisper"}

# Block 2: one language, three corpora, two noise maskers at the same SNR.
# White noise is spectrally flat; speech-shaped noise keeps the spectral
# envelope of the segment, so at equal SNR it masks the bands that carry the
# speech. The clean condition reuses the block 1 caches.
PLAN_BLOCK2 = [
    (corpus, "es", condition, 200)
    for corpus in ("fleurs", "voxpopuli", "mls")
    for condition in ("clean", "noise_snr5")
]

# Block 3: more languages, the conditions kept after block 2.
# MLS has no English config, VoxPopuli no Portuguese.
PLAN_BLOCK3 = [
    (corpus, lang, condition, 200)
    for corpus, langs in (("fleurs", ("es", "de", "fr", "en", "pt")),
                          ("voxpopuli", ("es", "de", "fr", "en")),
                          ("mls", ("es", "de", "fr", "pt")))
    for lang in langs
    for condition in ("clean", "noise_snr5")
]


def tag_of(corpus, lang, condition):
    """
    Build the directory tag of one (corpus, language, condition).

    Args:
        corpus: Corpus key.
        lang: Language code.
        condition: Condition name.

    Returns:
        Directory tag.
    """
    return f"{corpus}__{lang}__{condition}"


def plan_dirs(plan, drive_root=DRIVE_ROOT):
    """
    Resolve the output directory of every entry of a plan.

    Args:
        plan: List of (corpus, lang, condition, n_segments).
        drive_root: Persistent root directory.

    Returns:
        Output directory per tag.
    """
    return {tag_of(c, l, cond): os.path.join(drive_root, tag_of(c, l, cond))
            for c, l, cond, _ in plan}


def audio_usage(scratch_root=SCRATCH_ROOT):
    """
    Report how much Drive space the WAV cache occupies, per configuration.

    Args:
        scratch_root: Root directory of the WAV cache.

    Returns:
        Megabytes per tag and total.
    """
    usage, total = {}, 0
    if not os.path.isdir(scratch_root):
        return {"total_mb": 0.0}
    for tag in sorted(os.listdir(scratch_root)):
        directory = os.path.join(scratch_root, tag)
        if not os.path.isdir(directory):
            continue
        size = sum(os.path.getsize(os.path.join(directory, f))
                   for f in os.listdir(directory))
        usage[tag] = round(size / 1e6, 1)
        total += size
    usage["total_mb"] = round(total / 1e6, 1)
    return usage


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


def prepare(plan, drive_root=DRIVE_ROOT, scratch_root=SCRATCH_ROOT):
    """
    Stream and degrade every entry of a plan.

    Args:
        plan: List of (corpus, lang, condition, n_segments).
        drive_root: Persistent root directory.
        scratch_root: Local root directory for WAV files.

    Returns:
        Output directory per tag.
    """
    dirs = {}
    for corpus, lang, condition, n in plan:
        _, out_dir, stats = build_public_corpus(
            corpus, lang, n, drive_root, scratch_root,
            condition=condition, min_ref_words=0,
        )
        dirs[tag_of(corpus, lang, condition)] = out_dir
        print(f"\n=== {corpus} / {lang} / {condition} ===")
        print(json.dumps(stats, indent=1))
    return dirs


# --- transcription -----------------------------------------------------------


def run_step(kind, plan, systems=None, drive_root=DRIVE_ROOT,
             batch_size=8, chunk=128):
    """
    Run every role served by one model kind, over every entry of a plan.

    Args:
        kind: Model kind, one of RUNNERS.
        plan: List of (corpus, lang, condition, n_segments).
        systems: Role to (model id, kind) mapping.
        drive_root: Persistent root directory.
        batch_size: Batch size.
        chunk: Number of records between cache flushes.

    Returns:
        None.
    """
    systems = systems or SYSTEMS
    roles = roles_for(kind, systems)
    if not roles:
        print(f"no role uses kind {kind}, nothing to do")
        return

    runner = getattr(transcribe, RUNNERS[kind])
    for tag, out_dir in plan_dirs(plan, drive_root).items():
        records = read_records(os.path.join(out_dir, "records.jsonl"))
        missing = [r for r in records if not os.path.exists(r["wav_path"])]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} WAV files missing for {tag}, rerun prepare()"
            )
        for role in roles:
            model_id = systems[role][0]
            print(f"\n=== {tag} | {role}: {model_id} | {len(records)} segments ===")
            runner(records, os.path.join(out_dir, f"{role}.jsonl"),
                   base_model_path=model_id, batch_size=batch_size, chunk=chunk)


# --- pooling -----------------------------------------------------------------


def remap_results(results, triplet):
    """
    Reassign which transcribed system plays each role.

    Permuting the triplet is what isolates the effect of changing ASR systems
    from the effect of changing domain: corpus, language and condition stay
    fixed, only the roles move. No new inference is needed, the hypotheses are
    already cached.

    Args:
        results: Transcriptions per stored role.
        triplet: Stored role names, in slot order (target, proxy_a, proxy_b).

    Returns:
        Transcriptions keyed by slot name.
    """
    slots = ("target", "proxy_a", "proxy_b")
    if triplet is None:
        return results
    if len(triplet) != len(slots):
        raise ValueError(f"triplet must name {len(slots)} stored roles")
    missing = [role for role in triplet if role not in results]
    if missing:
        raise ValueError(f"no transcription cached for {missing}")
    return {slot: results[role] for slot, role in zip(slots, triplet)}


def pack_pool(plan, dest_root, drive_root=DRIVE_ROOT, systems=None):
    """
    Copy everything needed to rebuild a pool elsewhere: text only, no audio.

    The destination mirrors the source layout, so the internal machine only has
    to call load_pool(plan, drive_root=dest_root) to obtain the same labelled
    rows, and can then train any regressor locally. This removes the need to
    transport model weights at all.

    Args:
        plan: List of (corpus, lang, condition, n_segments).
        dest_root: Destination root directory.
        drive_root: Source root directory.
        systems: Role to (model id, kind) mapping.

    Returns:
        Copied file count and total size in megabytes.
    """
    import shutil

    systems = systems or SYSTEMS
    wanted = ["records.jsonl", "norm_config.json"] + \
             [f"{role}.jsonl" for role in systems]

    copied, total = 0, 0
    for tag, out_dir in plan_dirs(plan, drive_root).items():
        destination = os.path.join(dest_root, tag)
        os.makedirs(destination, exist_ok=True)
        for name in wanted:
            source = os.path.join(out_dir, name)
            if not os.path.exists(source):
                continue
            shutil.copy2(source, os.path.join(destination, name))
            copied += 1
            total += os.path.getsize(source)

    summary = {"files": copied, "size_mb": round(total / 1e6, 2),
               "dest_root": dest_root}
    print(json.dumps(summary, indent=1))
    return summary


def load_pool(plan, systems=None, drive_root=DRIVE_ROOT, target_role="target",
              triplet=None):
    """
    Merge the transcriptions of every entry of a plan into one labelled pool.

    Args:
        plan: List of (corpus, lang, condition, n_segments).
        systems: Role to (model id, kind) mapping.
        drive_root: Persistent root directory.
        target_role: Slot scored as the target system.
        triplet: Optional stored roles to place in (target, proxy_a, proxy_b),
            for the leave-one-system-out protocol.

    Returns:
        Pooled rows and coverage per tag.
    """
    systems = systems or SYSTEMS
    pool, coverage = [], {}

    for tag, out_dir in plan_dirs(plan, drive_root).items():
        records = read_records(os.path.join(out_dir, "records.jsonl"))
        results = {}
        for role in systems:
            cache = transcribe.load_cache(os.path.join(out_dir, f"{role}.jsonl"))
            if cache:
                results[role] = {sid: row["text"] for sid, row in cache.items()}
        results = remap_results(results, triplet)
        if target_role not in results:
            print(f"skipping {tag}: no {target_role} transcription yet")
            continue
        rows, meta = build_table(records, results, out_dir, target_role=target_role)
        pool.extend(rows)
        coverage[tag] = {"n_rows": len(rows), **meta["coverage"]}

    return pool, coverage


def wer_by_condition(pool):
    """
    Report the target WER of every acoustic condition.

    This is the block 2 exit criterion: the sweep is useful only if it produces
    a monotone and wide enough WER range.

    Args:
        pool: Pooled labelled rows.

    Returns:
        WER statistics per (corpus, condition).
    """
    import numpy as np

    buckets = {}
    for row in pool:
        key = (row.get("corpus", "?"), row.get("condition", "clean"))
        buckets.setdefault(key, []).append(row)

    table = {}
    for (corpus, condition), rows in sorted(buckets.items()):
        wer = np.array([r["label_wer"] for r in rows], dtype=float)
        errors = sum(r["label_errors"] for r in rows)
        ref = sum(r["n_ref_words"] for r in rows)
        table[f"{corpus}/{condition}"] = {
            "n": len(rows),
            "wer_word_weighted": round(errors / ref, 4) if ref else None,
            "wer_segment_mean": round(float(wer.mean()), 4),
            "wer_std": round(float(wer.std()), 4),
            "pct_zero_wer": round(float((wer == 0).mean()), 4),
        }
    return table


def proxy_blindness(pool, roles=("proxy_a", "proxy_b")):
    """
    Measure how often the proxies agree with the target while the target is wrong.

    Those segments carry no signal for a proxy-based estimator, so this is the
    share of the error the method cannot see, per condition.

    Args:
        pool: Pooled labelled rows.
        roles: Proxy ASR systems.

    Returns:
        Blind rate per (corpus, condition).
    """
    import numpy as np
    from scipy.stats import spearmanr

    from features import proxy_features

    buckets = {}
    for row in pool:
        key = (row.get("corpus", "?"), row.get("condition", "clean"))
        buckets.setdefault(key, []).append(row)

    table = {}
    for (corpus, condition), rows in sorted(buckets.items()):
        pwer = np.array([proxy_features(r, roles).get("pwer_mean", 0.0)
                         for r in rows], dtype=float)
        wer = np.array([r["label_wer"] for r in rows], dtype=float)
        table[f"{corpus}/{condition}"] = {
            "pct_pwer_zero": round(float((pwer == 0).mean()), 4),
            "pct_blind": round(float(((pwer == 0) & (wer > 0)).mean()), 4),
            "pearson_pwer_wer": round(float(np.corrcoef(pwer, wer)[0, 1]), 4)
            if pwer.std() > 0 else None,
            "spearman_pwer_wer": round(
                float(spearmanr(pwer, wer).statistic), 4)
            if pwer.std() > 0 else None,
        }
    return table


def length_report(pool, short_words=5, high_wer=1.0):
    """
    Expose the segments whose length makes their WER unbounded.

    WER divides by the number of reference words, so a three-word reference with
    two insertions scores above 0.6 and a one-word reference can score 5. Those
    points inflate RMSE and destroy Pearson correlations while leaving MAE and
    Spearman almost untouched, which is exactly the pattern to look for when the
    two families of metrics disagree.

    Args:
        pool: Pooled labelled rows.
        short_words: Reference length below which WER is considered unstable.
        high_wer: WER above which a segment is counted as extreme.

    Returns:
        Length and extreme-value statistics per (corpus, condition).
    """
    import numpy as np

    buckets = {}
    for row in pool:
        key = (row.get("corpus", "?"), row.get("condition", "clean"))
        buckets.setdefault(key, []).append(row)

    table = {}
    for (corpus, condition), rows in sorted(buckets.items()):
        words = np.array([r["n_ref_words"] for r in rows], dtype=float)
        wer = np.array([r["label_wer"] for r in rows], dtype=float)
        table[f"{corpus}/{condition}"] = {
            "n": len(rows),
            "ref_words_p05": int(np.percentile(words, 5)),
            "ref_words_median": int(np.median(words)),
            "pct_short_ref": round(float((words < short_words).mean()), 4),
            "wer_max": round(float(wer.max()), 3),
            "wer_p99": round(float(np.percentile(wer, 99)), 3),
            "pct_wer_above_1": round(float((wer > high_wer).mean()), 4),
            "share_of_sq_error_top1pct": round(float(
                np.sort(wer ** 2)[-max(1, len(wer) // 100):].sum()
                / max((wer ** 2).sum(), 1e-9)), 4),
        }
    return table


# --- evaluation --------------------------------------------------------------


def evaluate_pool(pool, protocols=("group", "lodo", "lolo", "loco"),
                  models=ep.MODELS, feature_set="transferable", n_boot=500):
    """
    Run every protocol on a pool and print the comparison tables.

    Args:
        pool: Pooled labelled rows.
        protocols: Protocols to run.
        models: Models to compare, from evaluate_public.MODELS.
        feature_set: "transferable" or "full".
        n_boot: Bootstrap replicates, 0 to skip.

    Returns:
        Summary per protocol.
    """
    summaries = {}
    for protocol in protocols:
        result = ep.run_protocol(pool, protocol=protocol, models=models,
                                 feature_set=feature_set)
        summary = ep.summarize(result, n_boot=n_boot)
        summary["folds"] = result["folds"]
        summaries[protocol] = summary
        print(f"\n=== protocol {protocol} ({feature_set}) ===")
        print(json.dumps({k: v for k, v in summary.items() if k != "folds"},
                         indent=1))
    return summaries


def save_report(payload, name, drive_root=DRIVE_ROOT):
    """
    Save a report as JSON next to the corpora.

    Args:
        payload: Serializable content.
        name: File name without extension.
        drive_root: Persistent root directory.

    Returns:
        Saved file path.
    """
    reports = os.path.join(drive_root, "reports")
    os.makedirs(reports, exist_ok=True)
    path = os.path.join(reports, name + ".json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False, default=str)
    return path


# --- usage -------------------------------------------------------------------
#
# Preamble, every session.
#
#   import sys
#   from google.colab import drive; drive.mount('/content/drive')
#   project_path = '/content/drive/MyDrive/Travail/WER_Predictor'
#   if project_path not in sys.path:
#       sys.path.append(project_path)
#
# ---------------------------------------------------------------- BLOCK 2 ----
#
# 2.0  Preparation. No GPU. Streams and degrades, ~2000 segments.
#
#   !pip install -q "datasets<4.0.0" soundfile librosa jiwer num2words scipy
#   import run_public as rp
#   rp.prepare(rp.PLAN_BLOCK2)
#
#   # Listen to one degraded segment before spending GPU time on 2000 of them:
#   from IPython.display import Audio
#   Audio('/content/wav_cache/fleurs__es__tel_noise_snr10/'
#         'fleurs__es__tel_noise_snr10__00000.wav')
#
# 2.1  Transcription, one step per model. Install, restart runtime, run.
#
#   !pip install -q -r {project_path}/requirements_parakeet.txt
#   import run_public as rp; rp.run_step("parakeet", rp.PLAN_BLOCK2)
#
#   !pip install -q -r {project_path}/requirements_canary.txt
#   import run_public as rp; rp.run_step("canary", rp.PLAN_BLOCK2)
#
#   !pip install -q -r {project_path}/requirements_whisper.txt
#   import run_public as rp; rp.run_step("whisper", rp.PLAN_BLOCK2, batch_size=4)
#
# 2.2  Data quality. No GPU.
#
#   import run_public as rp, json
#   pool, coverage = rp.load_pool(rp.PLAN_BLOCK2)
#   print(json.dumps(rp.wer_by_condition(pool), indent=1))
#   print(json.dumps(rp.proxy_blindness(pool), indent=1))
#   rp.save_report({"coverage": coverage,
#                   "wer": rp.wer_by_condition(pool),
#                   "blindness": rp.proxy_blindness(pool)}, "block2_data")
#
#   Keep a condition when it raises the WER without saturating it, and when
#   pct_blind drops relative to clean. Drop the others from PLAN_BLOCK3.
#
# 2.3  Performance on Spanish only, with the baselines. No GPU.
#
#   summaries = rp.evaluate_pool(pool, protocols=("group", "lodo", "loco"))
#   rp.save_report(summaries, "block2_transferable")
#
#   Models compared: mean and pwer are the baselines, then ridge, lasso, hgb
#   and xgb. To restrict the comparison:
#   rp.evaluate_pool(pool, protocols=("lodo",), models=("pwer", "ridge", "hgb"))
#   xgboost is optional; drop "xgb" from models if it is not installed.
#
#   No "lolo" here: block 2 is single-language. "loco" is the interesting one,
#   it says whether the model trained on some conditions holds on an unseen one.
#
# 2.4  Transfer to the internal Peruvian corpus. Two routes, no GPU.
#
#   Route A, copy the transcriptions and retrain internally. Preferred when any
#   regressor is wanted, not only a linear one: nothing but text moves.
#
#   rp.pack_pool(rp.PLAN_BLOCK2, "/content/drive/MyDrive/Travail/"
#                                "WER_Predictor/wer_public_pack")
#   # Copy that folder to the internal machine, then there:
#   import run_public as rp, evaluate_public as ep
#   INTERNAL_PACK = "<internal path>/wer_public_pack"
#   pool_public, _ = rp.load_pool(rp.PLAN_BLOCK2, drive_root=INTERNAL_PACK)
#   bundle = ep.fit_bundle(pool_public, model="hgb")   # or ridge, lasso, xgb
#
#   # The bundle has the shape deploy expects, so the existing code applies:
#   import pipeline as pl
#   from merge import read_table
#   rows_internal = read_table("<internal out_dir>")
#   ranking = pl.rank(bundle, rows_internal)
#
#   Route B, copy the coefficients only. Lighter, but linear models only.
#
#   import evaluate_public as ep, json
#   export = ep.ridge_export(pool)          # trained on public Spanish only
#   rp.save_report(export, "ridge_export_es")
#   print(json.dumps(export, indent=1))     # ~60 numbers, copy them internally
#
#   # Internal side, on the machine holding the labelled Peruvian table:
#   from merge import read_table
#   rows_internal = read_table("<internal out_dir>")
#   print(json.dumps(ep.transfer_report(export, rows_internal), indent=1))
#
#   # And the operational ranking, the replacement for pipeline.rank:
#   import pandas as pd
#   print(pd.DataFrame(ep.predict_calls_export(export, rows_internal))
#           .to_string(index=False))
#
#   Nothing is fitted internally and nothing is saved: no bundle, no
#   wer_model.joblib. apply_export is a dot product over the standardized
#   features, so the whole model travels as the JSON printed above.
#
#   Both routes answer the same question, so run whichever fits your
#   constraints; running both also tells whether a non-linear model buys
#   anything on internal data that the ridge does not.
#
#   Read transfer_report against its own baseline, not against your internal
#   numbers: what matters is ridge versus pwer on the same rows.
#
# 2.5  Leave-one-system-out, to make 2.4 interpretable. No GPU.
#
#   The internal triplet is the same as the public one, so 2.4 already varies
#   one thing only, the domain. This step is therefore no longer needed to
#   interpret it: it is a secondary robustness check, telling how much the
#   estimator depends on which system sits in which role. Run it if time
#   allows, after block 3.
#
#   import evaluate_public as ep, json
#   triplets = {
#       "parakeet_target": ("target", "proxy_a", "proxy_b"),
#       "canary_target":   ("proxy_a", "target", "proxy_b"),
#       "whisper_target":  ("proxy_b", "target", "proxy_a"),
#   }
#   loso = {}
#   for name, triplet in triplets.items():
#       pool_t, _ = rp.load_pool(rp.PLAN_BLOCK2, triplet=triplet)
#       loso[name] = ep.summarize(
#           ep.run_protocol(pool_t, protocol="group"), n_boot=0)
#   rp.save_report(loso, "block2_loso")
#
#   Train on one triplet, test on another (same corpus, same condition):
#   pool_a, _ = rp.load_pool(rp.PLAN_BLOCK2, triplet=triplets["parakeet_target"])
#   pool_b, _ = rp.load_pool(rp.PLAN_BLOCK2, triplet=triplets["canary_target"])
#   export_a = ep.ridge_export(pool_a)
#   print(json.dumps(ep.transfer_report(export_a, pool_b), indent=1))
#
#   Note that build_table rewrites table.jsonl in each corpus directory, so the
#   file on Drive holds whichever triplet ran last. The pool returned in memory
#   is the one that matters.
#
# ---------------------------------------------------------------- BLOCK 3 ----
#
# 3.0  Preparation and transcription: same three steps, with PLAN_BLOCK3.
#      Edit its condition tuple first to keep only what block 2 validated.
#
#   import run_public as rp; rp.prepare(rp.PLAN_BLOCK3)
#   ... then the three transcription steps with rp.PLAN_BLOCK3 ...
#
# 3.1  Protocols and baselines. No GPU, a few minutes.
#
#   import run_public as rp
#   pool, coverage = rp.load_pool(rp.PLAN_BLOCK3)
#   summaries = rp.evaluate_pool(pool)
#   rp.save_report(summaries, "block3_transferable")
#
#   # Same run with every feature, to measure what the scale-free
#   # restriction costs in distribution and gains out of it:
#   rp.save_report(rp.evaluate_pool(pool, feature_set="full"), "block3_full")
#
# 3.2  Export of the transferable ridge.
#
#   import evaluate_public as ep
#   export = ep.ridge_export(pool)
#   rp.save_report(export, "ridge_export")
#   # Internal side: apply_export(export, rows) is a dot product, no sklearn.
#
# Read the tables in this order:
#   1. models.pwer.MAE  -- the baseline to beat, Waheed et al. table 4
#   2. models.ridge.gain_vs_pwer  -- their regression gains about 38 percent
#   3. wer_std of each fold next to its MAE, since R2 depends on that spread
#   4. wer_corpus_gap  -- the aggregate bias, what the business actually reads
#   5. MAE_ci  -- if the intervals of ridge and pwer overlap, the gain is not
#      established on this sample size
