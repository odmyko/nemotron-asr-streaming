"""Text normalization shared by ordinary and speaker-tagged ASR."""
import re

_LANG_TAG_RE = re.compile(r"\s*<[a-zA-Z]{2,3}(?:-[a-zA-Z]{2,4})?>\s*")


def strip_lang_tags(text: str) -> str:
    return _LANG_TAG_RE.sub(" ", text).strip()
