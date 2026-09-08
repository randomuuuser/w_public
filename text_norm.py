"""
Utilities for text normalization and WER computation.

Provides a consistent normalization pipeline for reference and hypothesis
transcripts, along with helper functions for WER statistics.
"""

import re
import unicodedata

from num2words import num2words

# num2words language codes, keyed by the corpus language code
NUM2WORDS_LANG = {"es": "es", "de": "de", "en": "en", "fr": "fr"}

# Characters we keep: letters (incl. accents/umlauts/eszett), digits, spaces,
# and the apostrophe (meaningful in some languages).
_KEEP_APOSTROPHE = "'"

# A number token: optional sign, digit groups possibly separated by . or ,
# Examples matched: 5  1234  1.234  1,234  3,5  12.345,67  -7
_NUMBER_RE = re.compile(r"[-+]?\d[\d.,]*")

# German systems disagree on umlaut spelling ('möchte' vs 'moechte'), so the
# fold must transliterate (oe) rather than strip (o) to unify both forms.
_GERMAN_FOLD = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}


def _strip_accents(text, lang=None):
    """
    Normalize accented characters according to the target language.

    Args:
        text: Input text.
        lang: Language code.

    Returns:
        Normalized text.
    """
    if lang == "de":
        for source, target in _GERMAN_FOLD.items():
            text = text.replace(source, target)
    else:
        text = text.replace("ß", "ss")

    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _parse_number(raw):
    """
    Parse a numeric string into an integer or float.

    Args:
        raw: Numeric string.

    Returns:
        Parsed numeric value, or None if parsing fails.
    """
    sign = -1 if raw.startswith("-") else 1
    body = raw.lstrip("+-").rstrip(".,")
    if not body:
        return None

    seps = [c for c in body if c in ".,"]
    if not seps:
        return sign * int(body)

    last_sep = body.rfind(seps[-1])
    tail = body[last_sep + 1:]

    if len(tail) in (1, 2):
        integer_part = re.sub(r"[.,]", "", body[:last_sep])
        integer_part = integer_part or "0"
        return sign * float(f"{integer_part}.{tail}")

    return sign * int(re.sub(r"[.,]", "", body))


def _expand_numbers(text, lang):
    """
    Convert numeric expressions into their written form.
    This makes '25' and 'veinticinco' comparable.

    Args:
        text: Input text.
        lang: Language code.

    Returns:
        Text with expanded numbers.
    """
    n2w_lang = NUM2WORDS_LANG.get(lang, "en")

    def replace(match):
        value = _parse_number(match.group(0))
        if value is None:
            return " "
        try:
            return " " + num2words(value, lang=n2w_lang) + " "
        except (NotImplementedError, OverflowError, ValueError):
            # Fall back to reading digit by digit
            digits = re.sub(r"\D", "", match.group(0))
            return " " + " ".join(num2words(int(d), lang=n2w_lang) for d in digits) + " "

    return _NUMBER_RE.sub(replace, text)


def normalize_text(text, lang, strip_accents=False, expand_numbers=True):
    """
    Normalize a transcript before WER / pWER computation.

    Args:
        text: Input transcript.
        lang: Language code.
        strip_accents: Whether to remove diacritics.
        expand_numbers: Whether to expand numbers into words.

    Returns:
        A lowercase, punctuation-free, single-spaced string.
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()

    if expand_numbers:
        text = _expand_numbers(text, lang)

    if strip_accents:
        text = _strip_accents(text, lang)

    # Drop everything that is not a letter, a digit or an apostrophe
    text = "".join(
        c if (c.isalnum() or c == _KEEP_APOSTROPHE) else " " for c in text
    )

    return " ".join(text.split())


def wer_counts(reference, hypothesis):
    """
    Compute edit statistics and Word Error Rate.

    Args:
        reference: Normalized reference transcript.
        hypothesis: Normalized hypothesis transcript.

    Returns:
        Dictionary containing edit counts and WER.
    """
    import jiwer

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        n_ins = len(hyp_words)
        return {"S": 0, "D": 0, "I": n_ins, "n_ref": 0, "n_hyp": n_ins,
                "errors": n_ins, "wer": float(n_ins > 0)}

    out = jiwer.process_words(reference, hypothesis)
    errors = out.substitutions + out.deletions + out.insertions
    return {
        "S": out.substitutions,
        "D": out.deletions,
        "I": out.insertions,
        "n_ref": len(ref_words),
        "n_hyp": len(hyp_words),
        "errors": errors,
        "wer": errors / len(ref_words),
    }


def diagnose_accent_impact(pairs, lang):
    """
    Compare WER with and without accent normalization.

    Args:
        pairs: List of (reference, hypothesis) pairs.
        lang: Language code.

    Returns:
        WER for each normalization strategy.
    """
    totals = {"keep": [0, 0], "strip": [0, 0]}
    for reference, hypothesis in pairs:
        for mode, strip in (("keep", False), ("strip", True)):
            ref = normalize_text(reference, lang, strip_accents=strip)
            hyp = normalize_text(hypothesis, lang, strip_accents=strip)
            counts = wer_counts(ref, hyp)
            totals[mode][0] += counts["errors"]
            totals[mode][1] += counts["n_ref"]

    return {
        mode: (errs / n_ref if n_ref else 0.0)
        for mode, (errs, n_ref) in totals.items()
    }