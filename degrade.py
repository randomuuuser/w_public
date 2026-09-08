"""
Controlled acoustic degradations, applied to public audio before transcription.

Purpose: public test sets are clean read or parliamentary speech, so the WER of
a modern ASR system has almost no dispersion on them. Degrading the audio along
a physical axis produces a wide and, more importantly, *controlled* WER range,
and brings the acoustic conditions closer to telephone audio.

Provenance of the transforms:
  - additive noise, reverberation, echo, clipping, time stretch and pitch shift
    are the perturbation families used by Waheed et al. (Findings ACL 2025) to
    build their perturbed evaluation set;
  - the reverberation impulse response follows Polack's statistical model,
    exponentially decaying Gaussian noise parameterized by RT60;
  - the telephone channel is not a perturbation but a reproduction of a real
    one: 8 kHz narrowband plus ITU-T G.711 mu-law companding, which is what
    call recordings actually go through.

Every transform is deterministic given a seed, so a wiped Colab scratch is
regenerated identically.
"""

import hashlib

import numpy as np
from scipy.signal import fftconvolve, resample_poly

MU_LAW_MU = 255.0


def stable_seed(*parts):
    """
    Derive a reproducible seed from arbitrary strings.

    Args:
        parts: Strings identifying the segment and the condition.

    Returns:
        Integer seed.
    """
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8"))
    return int.from_bytes(digest.digest()[:4], "big")


def _peak_normalize(samples, peak=0.95):
    """
    Rescale samples so that they fit in [-1, 1] without hard clipping.

    Args:
        samples: Audio samples.
        peak: Target peak amplitude.

    Returns:
        Rescaled samples.
    """
    maximum = float(np.max(np.abs(samples))) if samples.size else 0.0
    if maximum > peak:
        samples = samples * (peak / maximum)
    return samples.astype(np.float32)


def add_noise(samples, sr, snr_db, rng):
    """
    Add white Gaussian noise at a target signal-to-noise ratio.

    Args:
        samples: Audio samples.
        sr: Sampling rate.
        snr_db: Target SNR in decibels.
        rng: Random generator.

    Returns:
        Noisy samples.
    """
    power = float(np.mean(samples ** 2))
    if power <= 0:
        return samples
    noise = rng.normal(0.0, 1.0, size=samples.shape)
    noise_power = float(np.mean(noise ** 2))
    scale = np.sqrt(power / (noise_power * 10.0 ** (snr_db / 10.0)))
    return _peak_normalize(samples + scale * noise)


def add_speech_shaped_noise(samples, sr, snr_db, rng):
    """
    Add noise whose spectrum matches the speech it masks.

    The masker keeps the magnitude spectrum of the segment and randomizes the
    phase, which destroys the temporal structure while preserving the spectral
    envelope. This is the classical speech-shaped noise construction used as a
    reference masker in speech intelligibility testing. At equal SNR it is more
    damaging than white noise, because it masks the bands that actually carry
    the speech energy instead of spreading power flat across the spectrum.

    Args:
        samples: Audio samples.
        sr: Sampling rate.
        snr_db: Target SNR in decibels.
        rng: Random generator.

    Returns:
        Noisy samples.
    """
    power = float(np.mean(samples ** 2))
    if power <= 0 or samples.size < 2:
        return samples

    magnitude = np.abs(np.fft.rfft(samples))
    phase = rng.uniform(-np.pi, np.pi, size=magnitude.shape)
    noise = np.fft.irfft(magnitude * np.exp(1j * phase), n=samples.size)

    noise_power = float(np.mean(noise ** 2))
    if noise_power <= 0:
        return samples
    scale = np.sqrt(power / (noise_power * 10.0 ** (snr_db / 10.0)))
    return _peak_normalize(samples + scale * noise)


def add_reverb(samples, sr, rt60, rng, length=0.5):
    """
    Convolve with a synthetic room impulse response.

    The impulse response is Gaussian noise with an exponential decay envelope
    reaching -60 dB at rt60 (Polack's statistical model).

    Args:
        samples: Audio samples.
        sr: Sampling rate.
        rt60: Reverberation time in seconds.
        rng: Random generator.
        length: Impulse response length in seconds.

    Returns:
        Reverberated samples.
    """
    n = int(length * sr)
    time = np.arange(n) / sr
    envelope = np.exp(-6.9078 * time / rt60)  # ln(1000) = 6.9078 -> -60 dB
    impulse = rng.normal(0.0, 1.0, size=n) * envelope
    impulse[0] = 1.0
    impulse = impulse / np.sqrt(np.sum(impulse ** 2))
    wet = fftconvolve(samples, impulse, mode="full")[:samples.size]
    return _peak_normalize(wet)


def add_echo(samples, sr, delay=0.15, decay=0.4):
    """
    Add a single delayed and attenuated copy of the signal.

    Args:
        samples: Audio samples.
        sr: Sampling rate.
        delay: Echo delay in seconds.
        decay: Echo amplitude relative to the direct signal.

    Returns:
        Samples with echo.
    """
    shift = int(delay * sr)
    echoed = np.copy(samples)
    if shift < samples.size:
        echoed[shift:] += decay * samples[:-shift]
    return _peak_normalize(echoed)


def clip_distort(samples, threshold=0.25):
    """
    Apply hard clipping, the saturation of an overdriven input stage.

    Args:
        samples: Audio samples.
        threshold: Clipping level.

    Returns:
        Distorted samples.
    """
    return _peak_normalize(np.clip(samples, -threshold, threshold) / threshold * 0.95)


def telephone(samples, sr, target_sr=8000):
    """
    Simulate a narrowband telephone channel with G.711 mu-law companding.

    Args:
        samples: Audio samples.
        sr: Sampling rate.
        target_sr: Channel sampling rate.

    Returns:
        Samples back at the original rate, having gone through the channel.
    """
    down = resample_poly(samples, target_sr, sr)

    # ITU-T G.711 mu-law compression, 8-bit quantization, expansion
    magnitude = np.log1p(MU_LAW_MU * np.abs(down)) / np.log1p(MU_LAW_MU)
    companded = np.sign(down) * magnitude
    quantized = np.round(companded * 127.0) / 127.0
    expanded = np.sign(quantized) * (
        (1.0 + MU_LAW_MU) ** np.abs(quantized) - 1.0
    ) / MU_LAW_MU

    return _peak_normalize(resample_poly(expanded, sr, target_sr))


def time_stretch(samples, sr, rate):
    """
    Change speaking rate without changing pitch.

    Args:
        samples: Audio samples.
        sr: Sampling rate.
        rate: Stretch factor, above 1 is faster.

    Returns:
        Stretched samples.
    """
    import librosa

    return _peak_normalize(librosa.effects.time_stretch(samples, rate=rate))


def pitch_shift(samples, sr, steps):
    """
    Shift pitch without changing duration.

    Args:
        samples: Audio samples.
        sr: Sampling rate.
        steps: Shift in semitones.

    Returns:
        Shifted samples.
    """
    import librosa

    return _peak_normalize(librosa.effects.pitch_shift(samples, sr=sr, n_steps=steps))


# Condition registry. Each entry is a list of (function, kwargs) applied in
# order. Names are the condition tags used in directory names, so keep them
# filesystem-safe and stable: renaming one invalidates its cached hypotheses.
CONDITIONS = {
    "clean": [],
    "noise_snr20": [(add_noise, {"snr_db": 20.0})],
    "noise_snr10": [(add_noise, {"snr_db": 10.0})],
    "noise_snr5": [(add_noise, {"snr_db": 5.0})],
    "noise_snr0": [(add_noise, {"snr_db": 0.0})],
    "ssn_snr10": [(add_speech_shaped_noise, {"snr_db": 10.0})],
    "ssn_snr5": [(add_speech_shaped_noise, {"snr_db": 5.0})],
    "ssn_snr0": [(add_speech_shaped_noise, {"snr_db": 0.0})],
    "reverb_rt60_600": [(add_reverb, {"rt60": 0.6})],
    "tel": [(telephone, {})],
    "tel_noise_snr10": [(telephone, {}), (add_noise, {"snr_db": 10.0})],
    "tel_noise_snr5": [(telephone, {}), (add_noise, {"snr_db": 5.0})],
    "echo": [(add_echo, {})],
    "distort": [(clip_distort, {})],
}


def apply_condition(samples, sr, condition, seed=0):
    """
    Apply a named condition to a segment.

    Args:
        samples: Audio samples.
        sr: Sampling rate.
        condition: Condition name from CONDITIONS.
        seed: Seed making the transform reproducible.

    Returns:
        Degraded samples.
    """
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition}, expected one of {list(CONDITIONS)}"
        )

    rng = np.random.default_rng(seed)
    out = np.asarray(samples, dtype=np.float32)
    for function, kwargs in CONDITIONS[condition]:
        if "rng" in function.__code__.co_varnames:
            out = function(out, sr, rng=rng, **kwargs)
        else:
            out = function(out, sr, **kwargs)
    return out
