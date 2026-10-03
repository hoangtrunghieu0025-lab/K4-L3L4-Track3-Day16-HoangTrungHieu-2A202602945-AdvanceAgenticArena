"""Hàm dùng chung cho `critic` và `citation_checker`.

Chuẩn hoá giống scorer (NFC + casefold + gộp khoảng trắng) và so khớp theo
từng DÒNG — để layer không xoá/đổi nhầm claim mà scorer vẫn chấp nhận, và
không giữ lại claim mà scorer sẽ chấm HALLUCINATED.
"""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")

#: Độ dài tối thiểu để scorer coi một trích dẫn là hợp lệ (MIN_SUPPORT_CHARS).
MIN_QUOTE_CHARS = 12


def norm(text) -> str:
    if not isinstance(text, str):
        return ""
    return _WS.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def norm_lines(text) -> list[str]:
    return [line for line in (norm(raw) for raw in str(text).splitlines()) if line]


def on_one_line(claim_text, lines) -> bool:
    """`claim_text` có nằm nguyên văn trong MỘT dòng nào của `lines` không?

    `lines` là danh sách dòng đã chuẩn hoá (xem `norm_lines`).
    """
    needle = norm(claim_text)
    if len(needle) < MIN_QUOTE_CHARS:
        return False
    return any(needle in line for line in lines)


def source_doc_id(ctx, claim_text, observed_lines, observed_text) -> str | None:
    """doc_id của tài liệu đã đọc mà một DÒNG của nó chứa `claim_text`.

    Ưu tiên tài liệu về nguyên vẹn từ một lần fetch sạch; nếu không có thì
    chấp nhận tài liệu mà doc_id đã xuất hiện trong quan sát (kết quả search
    hoặc bản fetch bị cắt — scorer vẫn coi là đã truy xuất).
    """
    corpus = getattr(ctx, "corpus", None)
    if corpus is None or not on_one_line(claim_text, observed_lines):
        return None
    fallback = None
    for doc in corpus.docs:
        if not on_one_line(claim_text, norm_lines(doc.body)):
            continue
        if doc.body in observed_text:
            return doc.doc_id
        if fallback is None and doc.doc_id in observed_text:
            fallback = doc.doc_id
    return fallback


#: Dấu câu / ký hiệu bọc ngoài mà mô hình thật hay thêm; cắt bỏ là cắt bớt hợp lệ.
_WRAP_CHARS = " 	\"'“”‘’*_`>•-–—.,;:!?()[]"


def trimmed(text: str) -> str:
    """`text` sau khi cắt dấu câu/ký hiệu bọc ngoài — vẫn là substring của `text`."""
    return text.strip(_WRAP_CHARS)
