"""
Build evaluation records from public ASR corpora (FLEURS, MLS, VoxPopuli).

Records follow the exact schema produced by prepare_dataset.read_segments, so
transcribe, merge, features, evaluate and deploy run unchanged on public data.

Audio is streamed from the Hugging Face Hub and written as mono 16 kHz WAV into
a local scratch directory (on Colab: /content, ephemeral, ~100 GB). Only the
JSONL records and hypotheses are persisted to Drive.

Only test splits are used: the train splits of these corpora were used to train
the target and proxy ASR systems.

Grouping. Two distinct fields, two distinct purposes:
  - group_id  : cross-validation group, always set. Prevents leakage between
                folds. On FLEURS it is a hash of the normalized reference,
                because the same FLoRes sentence is read by several speakers.
  - sample_id : recording session, only when the corpus provides a real one
                (MLS chapter, VoxPopuli speaker). Public corpora contain no
                calls, so segments are never merged into artificial ones: when
                no session exists, sample_id falls back to segment_id and
                session-level metrics are simply not reported.
"""

import hashlib
import os

import numpy as np

from degrade import apply_condition, stable_seed
from prepare_dataset import (DEFAULT_NORM, TARGET_SR, read_records,
                             write_norm_config, write_records, write_wav_mono)
from text_norm import normalize_text

# Corpus registry. "configs" maps our language code to the HF config name,
# "text_keys" lists reference fields in order of preference, "session_keys"
# lists fields that identify a real recording session.
CORPORA = {
    "fleurs": {
        "repo": "google/fleurs",
        "configs": {"en": "en_us", "es": "es_419", "de": "de_de",
                    "fr": "fr_fr", "pt": "pt_br"},
        "text_keys": ("raw_transcription", "transcription"),
        "session_keys": (),  # isolated read sentences, no session
    },
    "voxpopuli": {
        "repo": "facebook/voxpopuli",
        "configs": {"en": "en", "es": "es", "de": "de", "fr": "fr"},
        "text_keys": ("raw_text", "normalized_text"),
        "session_keys": ("speaker_id",),
    },
    "mls": {
        # No English config here: English MLS is LibriSpeech.
        "repo": "facebook/multilingual_librispeech",
        "configs": {"es": "spanish", "de": "german", "fr": "french",
                    "pt": "portuguese"},
        "text_keys": ("transcript", "text"),
        "session_keys": ("chapter_id", "speaker_id"),
    },
}


def corpus_config(corpus, lang):
    """
    Resolve the Hugging Face repository and config for a corpus and language.

    Args:
        corpus: Corpus key from CORPORA.
        lang: Language code.

    Returns:
        Repository id and config name.
    """
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus}, expected one of {list(CORPORA)}")
    spec = CORPORA[corpus]
    if lang not in spec["configs"]:
        raise ValueError(
            f"{corpus} has no config for {lang}, available: {list(spec['configs'])}"
        )
    return spec["repo"], spec["configs"][lang]


def has_sessions(corpus):
    """
    Whether a corpus exposes real recording sessions.

    Args:
        corpus: Corpus key from CORPORA.

    Returns:
        True when session-level aggregation is meaningful.
    """
    return bool(CORPORA[corpus]["session_keys"])


def _pick_text(example, keys):
    """
    Return the first non-empty reference field found in an example.

    Args:
        example: Dataset example.
        keys: Candidate field names.

    Returns:
        Reference text, empty when none is available.
    """
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _pick_session(example, keys):
    """
    Return the session identifier of an example.

    Args:
        example: Dataset example.
        keys: Candidate field names.

    Returns:
        Session value as a string, or None when the corpus has no session.
    """
    for key in keys:
        value = example.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _text_group(reference_norm):
    """
    Hash a normalized reference into a cross-validation group.

    FLEURS reads the same FLoRes sentence with several speakers, so grouping by
    text is what keeps identical references out of two different folds.

    Args:
        reference_norm: Normalized reference transcript.

    Returns:
        Short stable hash.
    """
    return "txt" + hashlib.sha1(reference_norm.encode("utf-8")).hexdigest()[:12]


def _audio_array(audio, sr=TARGET_SR):
    """
    Convert a datasets audio value into a mono float32 array.

    Handles both the dict layout (datasets < 4) and the decoder object layout
    (datasets >= 4, torchcodec).

    Args:
        audio: Audio value from a dataset example.
        sr: Expected sampling rate.

    Returns:
        Mono float32 samples.
    """
    if isinstance(audio, dict) and "array" in audio:
        samples = np.asarray(audio["array"], dtype=np.float32)
        rate = int(audio.get("sampling_rate", sr))
    elif hasattr(audio, "get_all_samples"):
        batch = audio.get_all_samples()
        samples = np.asarray(batch.data, dtype=np.float32)
        rate = int(batch.sample_rate)
    else:
        raise TypeError(f"unsupported audio value of type {type(audio)}")

    if samples.ndim > 1:
        axis = 0 if samples.shape[0] < samples.shape[1] else 1
        samples = samples.mean(axis=axis)
    if rate != sr:
        raise ValueError(f"expected {sr} Hz, got {rate} Hz: cast the audio column")
    return samples


def stream_records(corpus, lang, n, scratch_dir, split="test",
                   condition="clean", min_ref_words=0, norm_config=None,
                   shuffle_buffer=0, seed=0):
    """
    Stream a public corpus and export segments as WAV files plus records.

    Args:
        corpus: Corpus key from CORPORA.
        lang: Language code.
        n: Number of segments to keep.
        scratch_dir: Local directory for the WAV files.
        split: Dataset split, test only by design.
        condition: Acoustic condition tag, "clean" until degradations are added.
        min_ref_words: Minimum reference length.
        norm_config: Text normalization settings.
        shuffle_buffer: Streaming shuffle buffer, 0 to keep the native order.
        seed: Random seed for the shuffle.

    Returns:
        Segment records and streaming statistics.
    """
    from datasets import Audio, load_dataset

    spec = CORPORA[corpus]
    repo, config = corpus_config(corpus, lang)
    norm_config = dict(DEFAULT_NORM if norm_config is None else norm_config)
    os.makedirs(scratch_dir, exist_ok=True)
    prefix = f"{corpus}__{lang}__{condition}"

    dataset = load_dataset(repo, config, split=split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    if shuffle_buffer:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)

    records = []
    stats = {"seen": 0, "empty_text": 0, "too_short": 0, "no_session": 0,
             "kept": 0}

    for index, example in enumerate(dataset):
        if len(records) >= n:
            break
        stats["seen"] += 1

        text = _pick_text(example, spec["text_keys"])
        if not text:
            stats["empty_text"] += 1
            continue

        reference_norm = normalize_text(text, lang, **norm_config)
        n_ref_words = len(reference_norm.split())
        if n_ref_words < min_ref_words:
            stats["too_short"] += 1
            continue

        segment_id = f"{prefix}__{index:05d}"
        session = _pick_session(example, spec["session_keys"])
        if spec["session_keys"] and session is None:
            stats["no_session"] += 1

        # group_id drives GroupKFold, sample_id drives session-level aggregation
        if session is not None:
            group_id = f"{prefix}__{session}"
            sample_id = group_id
        else:
            group_id = f"{prefix}__{_text_group(reference_norm)}"
            sample_id = segment_id

        samples = _audio_array(example["audio"])
        if condition != "clean":
            samples = apply_condition(samples, TARGET_SR, condition,
                                      seed=stable_seed(segment_id, condition))
        duration = round(len(samples) / TARGET_SR, 3)
        wav_path = os.path.join(scratch_dir, segment_id + ".wav")
        if not os.path.exists(wav_path):
            write_wav_mono(wav_path, samples)

        records.append({
            "segment_id": segment_id,
            "sample_id": sample_id,
            "group_id": group_id,
            "session_id": session,
            "corpus": corpus,
            "condition": condition,
            "path_audio": None,
            "wav_path": wav_path,
            "speaker_role": "",  # public corpora have no operator/client role
            "speaker_id": example.get("speaker_id"),
            "start_time": 0.0,
            "end_time": duration,
            "duration": duration,
            "reference": text,
            "reference_norm": reference_norm,
            "n_ref_words": n_ref_words,
            "lang": lang,
        })
        stats["kept"] += 1

    return records, stats


def structure_stats(records):
    """
    Summarize the segment and session structure of a set of records.

    Args:
        records: Segment records.

    Returns:
        Counts and duration statistics.
    """
    if not records:
        return {"n_segments": 0}

    durations = {}
    for record in records:
        durations[record["sample_id"]] = \
            durations.get(record["sample_id"], 0.0) + record["duration"]

    sessions = np.array(list(durations.values()), dtype=float) / 60.0
    segments = np.array([r["duration"] for r in records], dtype=float)
    with_session = any(r["session_id"] is not None for r in records)

    return {
        "n_segments": len(records),
        "n_groups": len({r["group_id"] for r in records}),
        "n_sessions": len(durations) if with_session else 0,
        "hours": round(float(segments.sum()) / 3600.0, 3),
        "segment_seconds_mean": round(float(segments.mean()), 2),
        "segment_seconds_max": round(float(segments.max()), 2),
        "session_minutes_mean": round(float(sessions.mean()), 2)
            if with_session else None,
        "session_minutes_max": round(float(sessions.max()), 2)
            if with_session else None,
    }


def cached_corpus(out_dir, wav_dir, n):
    """
    Reuse an already prepared corpus instead of streaming it again.

    Streaming decodes every example even when the WAV files exist, which is the
    slow part; this skips it entirely. WAV paths are rebuilt from wav_dir, so
    the audio directory can be moved without invalidating the records.

    Args:
        out_dir: Corpus directory holding records.jsonl.
        wav_dir: Directory holding the WAV files.
        n: Number of segments required.

    Returns:
        Records when the corpus is complete on disk, None otherwise.
    """
    path = os.path.join(out_dir, "records.jsonl")
    if not os.path.exists(path):
        return None

    records = read_records(path)
    if len(records) < n:
        return None
    records = records[:n]

    for record in records:
        record["wav_path"] = os.path.join(wav_dir, record["segment_id"] + ".wav")
        if not os.path.exists(record["wav_path"]):
            return None
    return records


def build_public_corpus(corpus, lang, n, drive_root, scratch_root,
                        condition="clean", min_ref_words=0, norm_config=None,
                        shuffle_buffer=0, seed=0, refresh=False):
    """
    Prepare one public corpus: stream, export WAVs, save records.

    Args:
        corpus: Corpus key from CORPORA.
        lang: Language code.
        n: Number of segments to keep.
        drive_root: Persistent root directory (Drive).
        scratch_root: Local root directory for the WAV files.
        condition: Acoustic condition tag.
        min_ref_words: Minimum reference length.
        norm_config: Text normalization settings.
        shuffle_buffer: Streaming shuffle buffer.
        seed: Random seed.
        refresh: Stream again even when the corpus is already on disk.

    Returns:
        Records, output directory and preparation statistics.
    """
    tag = f"{corpus}__{lang}__{condition}"
    out_dir = os.path.join(drive_root, tag)
    scratch_dir = os.path.join(scratch_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    norm_config = dict(DEFAULT_NORM if norm_config is None else norm_config)
    write_norm_config(out_dir, norm_config)

    if not refresh:
        cached = cached_corpus(out_dir, scratch_dir, n)
        if cached is not None:
            write_records(cached, os.path.join(out_dir, "records.jsonl"))
            stats = {"cached": True, **structure_stats(cached),
                     "has_sessions": has_sessions(corpus)}
            return cached, out_dir, stats

    records, stats = stream_records(
        corpus, lang, n, scratch_dir, condition=condition,
        min_ref_words=min_ref_words, norm_config=norm_config,
        shuffle_buffer=shuffle_buffer, seed=seed,
    )
    write_records(records, os.path.join(out_dir, "records.jsonl"))

    stats.update(structure_stats(records))
    stats["has_sessions"] = has_sessions(corpus)
    return records, out_dir, stats
