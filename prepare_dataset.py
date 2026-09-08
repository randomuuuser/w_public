"""
Prepare the evaluation dataset by extracting speech segments, normalizing
references, and generating the files required for ASR inference.

Expected JSON shape (TranscriptionDataset Format):
    {"VERSION": ..., "samples": {
        "<sample_id>": {
            "sample_id": str,
            "path_audio": str,
            "channel_operator": <index or name>,
            "channel_client":   <index or name>,
            "prediction": {"segments": [...]},
            "label":      {"segments": [
                {"start_time": float, "end_time": float, "channel": ...,
                 "annotation": {"speaker_id": ..., "speaker_role": ...,
                                "text": str, "words": [...]}}
            ]}
        }, ...
    }}
"""

import json
import os
import subprocess
import wave

import numpy as np

from text_norm import normalize_text

TARGET_SR = 16000

DEFAULT_NORM = {"strip_accents": False, "expand_numbers": True}  # try "expand

NORM_CONFIG_NAME = "norm_config.json"


def write_norm_config(out_dir, norm_config):
    """
    Save the normalization configuration.

    Args:
        out_dir: Output directory.
        norm_config: Normalization settings.

    Returns:
        Path to the saved configuration.
    """
    path = os.path.join(out_dir, NORM_CONFIG_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(norm_config, handle, indent=1)
    return path


def read_norm_config(out_dir):
    """
    Load the normalization configuration.

    Args:
        out_dir: Output directory.

    Returns:
        Normalization settings.
    """
    path = os.path.join(out_dir, NORM_CONFIG_NAME)
    if not os.path.exists(path):
        return dict(DEFAULT_NORM)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# -----------------------------------------------------------------------------
# Audio I/O
# -----------------------------------------------------------------------------


def probe_channels(path):
    """
    Retrieve the number of audio channels.

    Args:
        path: Audio file path.

    Returns:
        Number of channels.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out.splitlines()[0])


def load_call(path, sr=TARGET_SR):
    """
    Load an audio recording.

    Args:
        path: Audio file path.
        sr: Target sampling rate.

    Returns:
        Audio samples.
    """
    n_channels = probe_channels(path)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(sr), "-"],
        capture_output=True, check=True,
    ).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio.reshape(-1, n_channels)


def write_wav_mono(path, samples, sr=TARGET_SR):
    """
    Save audio as a mono WAV file.

    Args:
        path: Output file path.
        samples: Audio samples.
        sr: Sampling rate.

    Returns:
        None.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


# -----------------------------------------------------------------------------
# Channel resolution
# -----------------------------------------------------------------------------


def _as_int(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        return int(value.strip())
    return None


def channel_offset(sample):
    declared = [_as_int(sample.get(key))
                for key in ("channel_operator", "channel_client")]
    declared = [d for d in declared if d is not None]
    return min(declared) if declared else 0


def resolve_channel(segment, sample, n_channels):
    if n_channels == 1:
        return None

    raw = segment.get("channel")
    numeric = _as_int(raw)
    if numeric is not None:
        index = numeric - channel_offset(sample)
        return index if 0 <= index < n_channels else None

    for position, key in enumerate(("channel_operator", "channel_client")):
        if str(raw) == str(sample.get(key)):
            declared = _as_int(sample.get(key))
            if declared is not None:
                index = declared - channel_offset(sample)
                return index if 0 <= index < n_channels else None
            return position if assume_order else None

    return None


# def resolve_channel(segment, sample, n_channels):
#     """
#     Resolve the channel associated with a speech segment.
#     ...


def extract_segment(call_audio, sr, start_time, end_time, channel_index,
                    pad=0.0):
    """
    Extract a speech segment from a recording.

    Args:
        call_audio: Audio samples.
        sr: Sampling rate.
        start_time: Segment start time.
        end_time: Segment end time.
        channel_index: Selected channel.
        pad: Padding duration.

    Returns:
        Extracted audio samples.
    """
    start = max(0, int((start_time - pad) * sr))
    end = min(call_audio.shape[0], int((end_time + pad) * sr))
    if end <= start:
        return np.zeros(0, dtype=np.float32)

    chunk = call_audio[start:end]
    if channel_index is None:
        return chunk.mean(axis=1)
    return chunk[:, channel_index]


# -----------------------------------------------------------------------------
# JSON parsing
# -----------------------------------------------------------------------------


def read_segments(json_path, lang, min_ref_words=10, roles=None,
                  source="label", norm_config=None):
    """
    Parse the dataset annotations into segment records.

    Args:
        json_path: Dataset annotation file.
        lang: Language code.
        min_ref_words: Minimum reference length.
        roles: Speaker roles to keep.
        source: Annotation source.
        norm_config: Text normalization settings.

    Returns:
        Records and parsing statistics.
    """
    norm_config = dict(DEFAULT_NORM if norm_config is None else norm_config)

    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)

    records = []
    stats = {"samples": 0, "segments": 0, "empty_text": 0,
             "too_short": 0, "bad_span": 0, "wrong_role": 0, "kept": 0}

    for sample_id, sample in data.get("samples", {}).items():
        stats["samples"] += 1
        segments = sample.get(source, {}).get("segments", []) or []

        for index, segment in enumerate(segments):
            stats["segments"] += 1
            annotation = segment.get("annotation") or {}
            text = (annotation.get("text") or "").strip()
            role = annotation.get("speaker_role")

            if roles is not None and role not in roles:
                stats["wrong_role"] += 1
                continue
            if not text:
                stats["empty_text"] += 1
                continue

            start_time = float(segment.get("start_time", 0.0))
            end_time = float(segment.get("end_time", 0.0))
            if end_time <= start_time:
                stats["bad_span"] += 1
                continue

            reference_norm = normalize_text(text, lang, **norm_config)
            n_ref_words = len(reference_norm.split())
            if n_ref_words < min_ref_words:
                stats["too_short"] += 1
                continue

            records.append({
                "segment_id": f"{sample_id}__{index:04d}",
                "sample_id": sample_id,
                "path_audio": sample.get("path_audio"),
                "channel_raw": segment.get("channel"),
                "channel_operator": sample.get("channel_operator"),
                "channel_client": sample.get("channel_client"),
                "speaker_role": role,
                "speaker_id": annotation.get("speaker_id"),
                "start_time": start_time,
                "end_time": end_time,
                "duration": round(end_time - start_time, 3),
                "reference": text,
                "reference_norm": reference_norm,
                "n_ref_words": n_ref_words,
                "lang": lang,
            })
            stats["kept"] += 1

    return records, stats


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------


def export_segments(records, out_dir, audio_root=""):
    """
    Export speech segments as WAV files.

    Args:
        records: Segment records.
        out_dir: Output directory.
        audio_root: Root audio directory.

    Returns:
        Exported records and export failures.
    """
    os.makedirs(out_dir, exist_ok=True)
    by_call = {}
    for record in records:
        by_call.setdefault(record["sample_id"], []).append(record)

    failures = []
    for sample_id, call_records in by_call.items():
        source_path = os.path.join(audio_root, call_records[0]["path_audio"])
        try:
            call_audio = load_call(source_path)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as error:
            failures.append((sample_id, repr(error)))
            continue

        n_channels = call_audio.shape[1]
        for record in call_records:
            wav_path = os.path.join(out_dir, record["segment_id"] + ".wav")
            if os.path.exists(wav_path):
                record["wav_path"] = wav_path
                continue

            channel_index = resolve_channel(
                {"channel": record["channel_raw"]},
                {"channel_operator": record["channel_operator"],
                 "channel_client": record["channel_client"]},
                n_channels,
            )
            samples = extract_segment(
                call_audio, TARGET_SR,
                record["start_time"], record["end_time"], channel_index,
            )
            if samples.size == 0:
                failures.append((record["segment_id"], "empty slice"))
                continue

            write_wav_mono(wav_path, samples)
            record["wav_path"] = wav_path
            record["channel_index"] = channel_index

    return [r for r in records if "wav_path" in r], failures


def write_nemo_manifest(records, manifest_path, task="asr", pnc="yes"):
    """
    Generate a NeMo ASR manifest.

    Args:
        records: Segment records.
        manifest_path: Output manifest path.
        task: Task type.
        pnc: Punctuation setting.

    Returns:
        Manifest path.
    """
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({
                "audio_filepath": os.path.abspath(record["wav_path"]),
                "duration": record["duration"],
                "taskname": task,
                "task": task,
                "source_lang": record["lang"],
                "target_lang": record["lang"],
                "pnc": pnc,
                "segment_id": record["segment_id"],
            }, ensure_ascii=False) + "\n")
    return manifest_path


def write_records(records, path):
    """
    Save segment records.

    Args:
        records: Segment records.
        path: Output file path.

    Returns:
        Output file path.
    """
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_records(path):
    """
    Load segment records.

    Args:
        path: Input file path.

    Returns:
        Segment records.
    """
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_corpus(json_path, lang, out_dir, audio_root="",
                   min_ref_words=10, roles=None, norm_config=None):
    """
    Prepare the complete dataset for transcription.

    Args:
        json_path: Dataset annotation file.
        lang: Language code.
        out_dir: Output directory.
        audio_root: Root audio directory.
        min_ref_words: Minimum reference length.
        roles: Speaker roles to keep.
        norm_config: Text normalization settings.

    Returns:
        Prepared records, processing statistics, and export failures.
    """
    os.makedirs(out_dir, exist_ok=True)
    norm_config = dict(DEFAULT_NORM if norm_config is None else norm_config)
    write_norm_config(out_dir, norm_config)

    records, stats = read_segments(
        json_path, lang, min_ref_words=min_ref_words, roles=roles,
        norm_config=norm_config,
    )
    records, failures = export_segments(records, os.path.join(out_dir, "wav"),
                                        audio_root)
    write_records(records, os.path.join(out_dir, "segments.jsonl"))
    write_nemo_manifest(records, os.path.join(out_dir, "manifest.json"))

    stats["exported"] = len(records)
    stats["failures"] = len(failures)
    total_hours = sum(r["duration"] for r in records) / 3600.0
    stats["hours"] = round(total_hours, 3)
    return records, stats, failures