"""Words, parameters and stop words shared by the tasks.

Every task splits text the same way: runs of letters (no digits or underscores), lowercased, at
least two characters long. Unlike gensim's simple_preprocess this keeps where each word is in the
text, so matches can be shown in the original.
"""

import re
from datetime import datetime, timezone

from fastapi import HTTPException

WORD = re.compile(r"[^\W\d_]{2,}")
DEFAULT_LANGUAGES = ["fi", "en"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD.finditer(text or "")]


def words_with_spans(text: str) -> tuple[list[str], list[int], list[int]]:
    """Words and the character span [start, end) of each."""
    found, starts, ends = [], [], []
    for m in WORD.finditer(text or ""):
        found.append(m.group(0).lower())
        starts.append(m.start())
        ends.append(m.end())
    return found, starts, ends


def stem(label: str | None) -> str:
    base = str(label or "file").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base


def int_param(params: dict, name: str, default: int | None, low: int | None = None, high: int | None = None) -> int | None:
    raw = params.get(name)
    if raw is None or str(raw).strip() == "" or str(raw).strip().lower() == "auto":
        return default
    try:
        value = int(float(str(raw).strip()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be a number") from exc
    if low is not None and value < low:
        raise HTTPException(status_code=400, detail=f"{name} must be at least {low}")
    if high is not None and value > high:
        raise HTTPException(status_code=400, detail=f"{name} must be at most {high}")
    return value


def float_param(params: dict, name: str, default: float, low: float = 0.0, high: float = 1.0) -> float:
    raw = params.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be a number") from exc
    if not low <= value <= high:
        raise HTTPException(status_code=400, detail=f"{name} must be between {low} and {high}")
    return value


def bool_param(params: dict, name: str, default: bool = False) -> bool:
    raw = params.get(name)
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def languages_from(params: dict) -> list[str]:
    """Checkbox list (`languages`) plus free codes (`language_codes`, e.g. "et, la")."""
    import stopwordsiso

    chosen = params.get("languages") or []
    if isinstance(chosen, str):
        chosen = [chosen]
    chosen = [str(c).strip().lower() for c in chosen if str(c).strip()]
    chosen += [c.strip().lower() for c in re.split(r"[,\s]+", str(params.get("language_codes") or "")) if c.strip()]
    if not chosen:
        chosen = list(DEFAULT_LANGUAGES)
    unknown = [c for c in chosen if not stopwordsiso.has_lang(c)]
    if unknown:
        raise HTTPException(status_code=400, detail=f"No stop-word list for: {', '.join(unknown)}")
    return sorted(set(chosen))


def stop_words(languages: list[str], extra: str) -> set[str]:
    import stopwordsiso

    found = {w.lower() for w in stopwordsiso.stopwords(languages)}
    found |= {w.strip().lower() for w in re.split(r"[,\n]+", extra or "") if w.strip()}
    return found
