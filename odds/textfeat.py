"""Counts computed from a question's title and body (the body is HTML).

Used unchanged by training and by the API, so the two cannot disagree.
"""
import re

TEXT_FEATURES = [
    "title_chars", "title_words", "title_has_qmark", "title_upper_frac",
    "body_chars", "body_words", "n_paragraphs", "n_code_blocks", "n_inline_code",
    "code_frac", "n_links", "n_images", "n_list_items", "mentions_error", "mentions_version",
]

_TAG = re.compile(r"<[^>]+>")
_PRE = re.compile(r"<pre\b.*?</pre>", re.S | re.I)
_ERROR = re.compile(r"\b(error|fail(s|ed|ure)?|cannot|can't|unable|crash(es|ed)?|broken|not working)\b", re.I)
_VERSION = re.compile(r"\b\d{2}\.\d{2}\b")          # 22.04, 18.10 ...


def strip_html(html: str) -> str:
    return _TAG.sub(" ", html or "")


def text_features(title, body) -> dict:
    title = title or ""
    body = body or ""
    lower = body.lower()
    pre_blocks = _PRE.findall(body)
    code_chars = sum(len(strip_html(b)) for b in pre_blocks)
    plain = strip_html(body)
    letters = [c for c in title if c.isalpha()]
    both = title + " " + plain
    return {
        "title_chars": len(title),
        "title_words": len(title.split()),
        "title_has_qmark": int("?" in title),
        "title_upper_frac": (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        "body_chars": len(plain),
        "body_words": len(plain.split()),
        "n_paragraphs": lower.count("<p>"),
        "n_code_blocks": len(pre_blocks),
        "n_inline_code": max(lower.count("<code>") - len(pre_blocks), 0),
        "code_frac": code_chars / len(plain) if plain else 0.0,
        "n_links": lower.count("<a "),
        "n_images": lower.count("<img"),
        "n_list_items": lower.count("<li>"),
        "mentions_error": int(bool(_ERROR.search(both))),
        "mentions_version": int(bool(_VERSION.search(both))),
    }


def document(title, body, max_chars: int = 3000) -> str:
    """Title plus plain-text body, for the text-only model."""
    return ((title or "") + " \n " + strip_html(body))[:max_chars]
