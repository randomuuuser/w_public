"""
Run ASR systems on the prepared speech segments and store their transcriptions
for downstream WER estimation.
"""

import gc
import json
import os
import torch
from collections import defaultdict
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

WHISPER_DIR = "/domino/datasets/ModelHub-model-huggingface-openai/whisper-large-v3"
CANARY_1B_V2_DIR = "/domino/datasets/ModelHub-model-huggingface-nvidia/canary-1b-v2/main/canary-1b-v2.nemo"
PARAKEET_DIR = "/domino/datasets/ModelHub-model-huggingface-nvidia/parakeet-tdt-0.6b-v3/main/parakeet-tdt-0.6b-v3.nemo"
CANARY_180M_FLASH = "/domino/datasets/ModelHub-model-huggingface-nvidia/canary-180m-flash/main/canary-180m-flash.nemo"
CANARY_180M_FLASH_FT = "/domino/datasets/canary-180m-flash-finetuned/canary-180m-flash-finetune.nemo"
SAMPLING_RATE = 16_000


def _is_local(path):
    """True when path points at a local checkpoint rather than a HF repo id."""
    return os.path.exists(path)


def load_cache(path):
    """
    Load a transcription cache.

    Args:
        path: Cache file path.

    Returns:
        Cached transcriptions. Format : {segment_id: record}
    """
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return {r["segment_id"]: r for r in map(json.loads, handle) if r}


def append_cache(path, rows):
    """
    Append transcriptions to a cache.

    Args:
        path: Cache file path.
        rows: Transcription records.

    Returns:
        None.
    """
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def pending(records, cache):
    """
    Retrieve records that have not yet been transcribed.

    Args:
        records: Segment records.
        cache: Existing cache.

    Returns:
        Pending records.
    """
    return [r for r in records if r["segment_id"] not in cache]


def release_model(model):
    """
    Release model resources.

    Args:
        model: Loaded ASR model.

    Returns:
        None.
    """
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_canary(base_model_path=CANARY_1B_V2_DIR, beam_size=1):
    """
    Load a Canary ASR model.

    Args:
        model_name: Model identifier.
        beam_size: Decoding beam size.

    Returns:
        Loaded model.
    """
    from nemo.collections.asr.models import EncDecMultiTaskModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if _is_local(base_model_path):
        model = EncDecMultiTaskModel.restore_from(base_model_path, map_location=device)
    else:
        model = EncDecMultiTaskModel.from_pretrained(model_name="nvidia/canary-1b-v2", map_location=device)
    model.eval()
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = beam_size
    model.change_decoding_strategy(decode_cfg)
    return model


def load_parakeet(base_model_path=PARAKEET_DIR, beam_size=1, dtype=None):
    from nemo.collections.asr.models import ASRModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _is_local(base_model_path):
        model = ASRModel.restore_from(base_model_path, map_location=device)
    else:
        model = ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3", map_location=device)
    
    if dtype is None:
        major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
        dtype = torch.bfloat16 if major >= 8 else torch.float32
    model.to(dtype)
    model.eval()
    return model


def hypothesis_text(item):
    """
    Extract the transcription text from a model output.

    Args:
        item: Model output.

    Returns:
        Transcription text.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)) and item:
        return hypothesis_text(item[0])
    return getattr(item, "text", "") or ""

def as_hypothesis_list(outputs, expected):
    """
    Normalize a NeMo transcribe() return value into a flat list.

    Depending on the version and the decoding strategy, transcribe() returns a
    flat list of hypotheses or a (best, all) tuple.

    Args:
        outputs: Raw return value of model.transcribe().
        expected: Number of segments sent in this batch.

    Returns:
        One hypothesis per segment.
    """
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if len(outputs) != expected and len(outputs) == 1:
        outputs = outputs[0]  # nested [[hyp, hyp, ...]]
    if len(outputs) != expected:
        raise ValueError(f"expected {expected} hypotheses, got {len(outputs)}")
    return outputs


def run_canary(records, cache_path, base_model_path=CANARY_1B_V2_DIR,
               batch_size=16, chunk=256):
    """
    Transcribe speech segments with a Canary model.

    Args:
        records: Segment records.
        cache_path: Cache file path.
        model_name: Model identifier.
        batch_size: Batch size.
        chunk: Processing chunk size.

    Returns:
        Updated transcription cache.
    """
    cache = load_cache(cache_path)
    todo = pending(records, cache)
    if not todo:
        return cache

    model = load_canary(base_model_path)
    manifest_path = cache_path + ".tmp_manifest.json"

    try:
        for start in range(0, len(todo), chunk):
            batch = todo[start:start + chunk]
            with open(manifest_path, "w", encoding="utf-8") as handle:
                for record in batch:
                    handle.write(json.dumps({
                        "audio_filepath": os.path.abspath(record["wav_path"]),
                        "duration": record["duration"],
                        "taskname": "asr",
                        "task": "asr",
                        "source_lang": record["lang"],
                        "target_lang": record["lang"],
                        "pnc": "yes",
                    }, ensure_ascii=False) + "\n")

            outputs = model.transcribe(manifest_path, batch_size=batch_size)
            rows = [
                {"segment_id": record["segment_id"],
                 "system": base_model_path,
                 "text": hypothesis_text(item)}
                for record, item in zip(batch, outputs)
            ]
            
            append_cache(cache_path, rows)
            for row in rows:
                cache[row["segment_id"]] = row
            print(f"  {base_model_path}: {min(start + chunk, len(todo))}/{len(todo)}")
    finally:
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        release_model(model)

    return cache


def run_parakeet(records, cache_path, base_model_path=PARAKEET_DIR,
                 batch_size=16, chunk=256):
    """
    Transcribe speech segments with a Parakeet model.

    Args:
        records: Segment records.
        cache_path: Cache file path.
        model_name: Model identifier.
        batch_size: Batch size.
        chunk: Processing chunk size.

    Returns:
        Updated transcription cache.
    """
    cache = load_cache(cache_path)
    todo = pending(records, cache)
    if not todo:
        return cache

    model = load_parakeet(base_model_path)
    manifest_path = cache_path + ".tmp_manifest.json"

    try:
        for start in range(0, len(todo), chunk):
            batch = todo[start:start + chunk]
            with open(manifest_path, "w", encoding="utf-8") as handle:
                for record in batch:
                    handle.write(json.dumps({
                        "audio_filepath": os.path.abspath(record["wav_path"]),
                        "duration": record["duration"],
                        "taskname": "asr",
                        "task": "asr",
                        "source_lang": record["lang"],
                        "target_lang": record["lang"],
                        "pnc": "yes",
                    }, ensure_ascii=False) + "\n")

            outputs = model.transcribe(manifest_path, batch_size=batch_size)
            outputs = as_hypothesis_list(outputs, len(batch))
            rows = [
                {"segment_id": record["segment_id"],
                 "system": base_model_path,
                 "text": hypothesis_text(item)}
                for record, item in zip(batch, outputs)
            ]
            append_cache(cache_path, rows)
            for row in rows:
                cache[row["segment_id"]] = row
            print(f"  {base_model_path}: {min(start + chunk, len(todo))}/{len(todo)}")
    finally:
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        release_model(model)

    return cache


def sanity_check_canary(records, base_model_path=CANARY_1B_V2_DIR,
                        n_per_lang=20):
    """
    Run a quick transcription check on a subset of segments.

    Args:
        records: Segment records.
        model_name: Model identifier.
        n_per_lang: Number of segments per language.

    Returns:
        Sample transcription cache.
    """
    from collections import Counter

    sample = []
    per_lang = Counter()
    for record in records:
        if per_lang[record["lang"]] < n_per_lang:
            sample.append(record)
            per_lang[record["lang"]] += 1

    cache_path = "sanity_canary.jsonl"
    if os.path.exists(cache_path):
        os.remove(cache_path)
    cache = run_canary(sample, cache_path, base_model_path=base_model_path)

    for record in sample:
        hypothesis = cache.get(record["segment_id"], {}).get("text", "")
        print(f"\n[{record['lang']}] {record['segment_id']}")
        print(f"  REF : {record['reference']}")
        print(f"  HYP : {hypothesis}")

    return cache


# -----------------------------------------------------------------------------
# Whisper Model
# -----------------------------------------------------------------------------


def whisper_lang(lang):
    """
    Map a record language string to a Whisper language token.

    Args:
        lang: The language string from a record (e.g., "es-PE", "en_US").

    Returns:
        str: The normalized language token (e.g., "es"), or None if auto-detection is required.
    """
    if not lang:
        return None
    return lang.split("-")[0].split("_")[0].lower()  # "es-PE" -> "es"


def load_whisper(base_model_path=WHISPER_DIR, device=None):
    """
    Load the Whisper-large-v3 model from a local folder.

    Args:
        base_model_path: Path to the local Whisper model directory.
        device: Target device for inference ('cuda:0', 'cpu', etc.). Defaults to available GPU or CPU.

    Returns:
        tuple: A pair containing the WhisperProcessor and WhisperForConditionalGeneration model in eval mode.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    if _is_local(base_model_path):
        processor = WhisperProcessor.from_pretrained(base_model_path, local_files_only=True)      
    else:
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", local_files_only=_is_local(base_model_path))    

    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=dtype,  # try dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        local_files_only=_is_local(base_model_path),
    )
    model.to(device)
    model.eval()

    # Legacy checkpoints ship forced_decoder_ids, which conflicts with language=/task=
    model.generation_config.forced_decoder_ids = None
    return processor, model


def read_audio(wav_path):
    """
    Load an audio file and normalize it to mono 16 kHz float32.

    Args:
        wav_path: Filesystem path to the input audio file.

    Returns:
        np.ndarray: The processed audio waveform as a float32 array.
    """
    audio, _ = librosa.load(wav_path, sr=SAMPLING_RATE, mono=True)
    return audio


def transcribe_short(processor, model, audios, lang):
    """
    Perform batched decoding for short audio segments (under 30 seconds).

    Args:
        processor: The WhisperProcessor instance.
        model: The loaded Whisper model.
        audios: List of pre-processed audio waveforms.
        lang: The target language token.

    Returns:
        list: A list of decoded text strings corresponding to the input audios.
    """
    inputs = processor(audios, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    inputs = inputs.to(model.device, model.dtype)
    with torch.inference_mode():
        tokens = model.generate(
            **inputs,
            language=lang,
            task="transcribe",
            num_beams=1,
            return_timestamps=False,
        )
    return [t.strip() for t in processor.batch_decode(tokens, skip_special_tokens=True)]


def transcribe_long(processor, model, audio, lang):
    """
    Perform sequential long-form decoding for audio segments over 30 seconds.

    Args:
        processor: The WhisperProcessor instance.
        model: The loaded Whisper model.
        audio: A single pre-processed audio waveform.
        lang: The target language token.

    Returns:
        str: The decoded transcription text for the long-form segment.
    """
    inputs = processor(
        audio,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
    )
    inputs = inputs.to(model.device, model.dtype)
    with torch.inference_mode():
        tokens = model.generate(
            **inputs,
            language=lang,
            task="transcribe",
            num_beams=1,
            return_timestamps=True,  # mandatory for long-form
            condition_on_prev_tokens=False,  # no context leak across windows
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            logprob_threshold=-1.0,
            compression_ratio_threshold=1.35,
            no_speech_threshold=0.6,
        )
    return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def run_whisper(records, cache_path, base_model_path=WHISPER_DIR, batch_size=8, chunk=256):
    """
    Transcribe every pending record using a local Whisper checkpoint.

    Args:
        records: List of segment records containing metadata and file paths.
        cache_path: Path to the cache file for storing results.
        base_model_path: Path to the local Whisper model directory.
        batch_size: Number of segments to process in parallel.
        chunk: Number of records to process before saving progress.

    Returns:
        dict: The updated transcription cache dictionary.
    """
    cache = load_cache(cache_path)
    todo = pending(records, cache)
    if not todo:
        return cache

    system = base_model_path
    processor, model = load_whisper(base_model_path)
    try:
        for start in range(0, len(todo), chunk):
            block = todo[start:start + chunk]
            rows = []

            by_lang = defaultdict(list)
            for record in block:
                by_lang[whisper_lang(record["lang"])].append(record)

            for lang, group in by_lang.items():
                short = [r for r in group if r["duration"] <= 30.0]
                long = [r for r in group if r["duration"] > 30.0]

                for i in range(0, len(short), batch_size):
                    sub = short[i:i + batch_size]
                    audios = [read_audio(r["wav_path"]) for r in sub]
                    texts = transcribe_short(processor, model, audios, lang)
                    rows += [
                        {"segment_id": r["segment_id"], "system": system, "text": t}
                        for r, t in zip(sub, texts)
                    ]

                for record in long:
                    text = transcribe_long(
                        processor, model, read_audio(record["wav_path"]), lang
                    )
                    rows.append(
                        {"segment_id": record["segment_id"], "system": system, "text": text}
                    )

            append_cache(cache_path, rows)
            for row in rows:
                cache[row["segment_id"]] = row
            print(f"  {system}: {min(start + chunk, len(todo))}/{len(todo)}")
    finally:
        del processor
        release_model(model)
    return cache


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

# Peru
SYSTEMS = {
    "target": (PARAKEET_DIR, "parakeet"),
    "proxy_a": (CANARY_1B_V2_DIR, "canary"),
    "proxy_b": (WHISPER_DIR, "whisper"),
}


def run_all(records, out_dir, systems=None):
    """
    Run all configured ASR systems.

    Args:
        records: Segment records.
        out_dir: Output directory.
        systems: ASR systems to execute.

    Returns:
        Transcriptions grouped by system.
    """
    os.makedirs(out_dir, exist_ok=True)
    systems = systems or SYSTEMS
    results = {}

    for role, (base_model_path, kind) in systems.items():
        cache_path = os.path.join(out_dir, f"{role}.jsonl")
        print(f"\n=== {role}: {base_model_path} ===")
        if kind == "canary":
            runner = run_canary
        elif kind == "whisper":
            runner = run_whisper
        else:
            runner = run_parakeet
        cache = runner(records, cache_path, base_model_path=base_model_path)
        results[role] = {sid: row["text"] for sid, row in cache.items()}

    return results