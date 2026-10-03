"""Hàm dùng chung cho các layer.

Mọi phép so khớp chạy trên dạng chuẩn hoá GIỐNG scorer (NFC + casefold +
gộp khoảng trắng) và theo từng DÒNG tài liệu — để layer không xoá/đổi nhầm
claim mà scorer vẫn chấp nhận, và không giữ claim mà scorer sẽ chấm hỏng.

Không hàm nào ở đây đọc `Doc.tags` hay biết brief nào đang chạy: mọi tín hiệu
đều lấy từ CHỮ của tài liệu và của báo cáo.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)
_DOC_ID = re.compile(r"doc[\s_-]*0*(\d{1,4})", re.IGNORECASE)

#: Độ dài tối thiểu để scorer coi một trích dẫn là hợp lệ (MIN_SUPPORT_CHARS).
MIN_QUOTE_CHARS = 12

#: Ngân sách mặc định của scorer khi brief không đặt (arena.scorer.DEFAULT_BUDGET).
DEFAULT_MAX_TOOL_CALLS = 8
DEFAULT_MAX_TOKENS = 12000

#: Dấu câu / ký hiệu bọc ngoài mà mô hình thật hay thêm; cắt bỏ là cắt bớt hợp lệ.
_WRAP_CHARS = " \t\r\n\"'“”‘’«»*_`>•·-–—.,;:!?()[]{}…"

#: Tài liệu TỰ NHẬN là đã hết hiệu lực / chỉ áp dụng hẹp / chỉ là tổng hợp:
#: không phải nguồn có thẩm quyền ngang hàng, nên không tạo "mâu thuẫn" thật.
_NON_AUTHORITATIVE = (
    "lỗi thời",
    "không còn hiệu lực",
    "hết hiệu lực",
    "đã được thay thế",
    "thí điểm",
    "không áp dụng toàn quốc",
    "chỉ áp dụng cho chi nhánh",
    "không thay thế cho văn bản chính sách",
    "không thay thế chính sách",
)

#: Câu nói rằng dữ liệu KHÔNG có / chưa có — tín hiệu của brief "absent".
_INSUFFICIENT = (
    "chưa được đồng bộ",
    "không có số liệu",
    "chưa có số liệu",
    "không có dữ liệu",
    "chưa có dữ liệu",
    "không đủ dữ liệu",
    "không đủ căn cứ",
    "không đủ bằng chứng",
    "không suy diễn",
    "không tìm thấy thông tin",
    "không có thông tin",
)


def norm(text) -> str:
    if not isinstance(text, str):
        return ""
    return _WS.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


@lru_cache(maxsize=4096)
def _lines_of(text: str) -> tuple:
    return tuple(line for line in (norm(raw) for raw in text.splitlines()) if line)


def norm_lines(text) -> tuple:
    return _lines_of(text if isinstance(text, str) else str(text))


def on_one_line(claim_text, lines) -> bool:
    """`claim_text` có nằm nguyên văn trong MỘT dòng nào của `lines` không?"""
    needle = norm(claim_text)
    if len(needle) < MIN_QUOTE_CHARS:
        return False
    return any(needle in line for line in lines)


def trimmed(text: str) -> str:
    """`text` sau khi cắt dấu câu/ký hiệu bọc ngoài — vẫn là substring của `text`."""
    return text.strip(_WRAP_CHARS)


def words(text) -> list:
    return _WORD.findall(norm(text))


def numbers(text) -> frozenset:
    return frozenset(w for w in words(text) if any(ch.isdigit() for ch in w))


def content_words(text) -> frozenset:
    return frozenset(
        w for w in words(text) if len(w) > 1 and not any(ch.isdigit() for ch in w)
    )


def family(doc) -> str:
    """Chủ đề của tài liệu: phần tiêu đề trước " — " (vd. "an toàn lao động tại kho")."""
    title = getattr(doc, "title", "") or ""
    return norm(title.split(" — ")[0])


def non_authoritative(doc) -> bool:
    body = norm(getattr(doc, "body", "")) + " " + norm(getattr(doc, "title", ""))
    return any(marker in body for marker in _NON_AUTHORITATIVE)


def says_insufficient(text) -> bool:
    text = norm(text)
    return any(marker in text for marker in _INSUFFICIENT)


def max_tool_calls(ctx):
    value = ctx.max_tool_calls
    return DEFAULT_MAX_TOOL_CALLS if value is None else value


def max_tokens(ctx):
    budget = ctx.budget if isinstance(ctx.budget, dict) else {}
    value = budget.get("max_tokens")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return DEFAULT_MAX_TOKENS
    return value


def canonical_doc_id(raw, corpus):
    """"DOC-4", "doc_0004", "[doc-0004]" -> "doc-0004" nếu tài liệu đó tồn tại."""
    if not isinstance(raw, str) or corpus is None:
        return raw
    if corpus.get(raw) is not None:
        return raw
    match = _DOC_ID.search(raw)
    if match:
        candidate = f"doc-{int(match.group(1)):04d}"
        if corpus.get(candidate) is not None:
            return candidate
    return raw


# ---------------------------------------------------------------------------
# Tài liệu đã truy xuất — mô phỏng đúng cách scorer tính `retrieved`.
# ---------------------------------------------------------------------------


def retrieved_ids(ctx) -> set:
    """doc_id mà scorer sẽ coi là "đã truy xuất".

    Nguồn chính là `ctx.state["retrieved"]`, do `citation_checker` ghi lại
    trong `wrap_tool_call` (fetch_doc + phát lại mỗi search). Cộng thêm các
    tài liệu có toàn văn nằm trong quan sát, phòng khi layer kia không chạy.
    """
    state = ctx.state if isinstance(getattr(ctx, "state", None), dict) else {}
    found = set(state.get("retrieved", ()))
    corpus = getattr(ctx, "corpus", None)
    if corpus is not None:
        observed = ctx.observed_text
        found.update(d.doc_id for d in corpus.docs if d.body and d.body in observed)
    return found


def source_doc_id(ctx, claim_text, prefer=None, retrieved=None):
    """doc_id của tài liệu ĐÃ TRUY XUẤT mà một DÒNG của nó chứa `claim_text`.

    Ưu tiên: tài liệu đang được trích (`prefer`) -> tài liệu có toàn văn trong
    quan sát -> tài liệu đã truy xuất khác. Không có -> None (đừng bịa doc_id).
    """
    corpus = getattr(ctx, "corpus", None)
    if corpus is None or len(norm(claim_text)) < MIN_QUOTE_CHARS:
        return None
    retrieved = retrieved_ids(ctx) if retrieved is None else retrieved
    observed = ctx.observed_text
    best, best_rank = None, 9
    for doc in corpus.docs:
        if doc.doc_id not in retrieved or not on_one_line(claim_text, norm_lines(doc.body)):
            continue
        rank = 0 if doc.doc_id == prefer else (1 if doc.body in observed else 2)
        if rank < best_rank:
            best, best_rank = doc.doc_id, rank
    return best


# ---------------------------------------------------------------------------
# Chuẩn hoá cấu trúc báo cáo (KHÔNG đụng tới chữ của claim).
# ---------------------------------------------------------------------------

_TEXT_KEYS = ("text", "quote", "claim", "sentence", "content", "evidence")
_DOC_KEYS = ("doc_id", "docid", "docId", "doc", "document", "source", "citation", "id")


def normalise_report(ctx, report) -> None:
    """Sửa HÌNH DẠNG báo cáo mà mô hình thật hay viết lệch, tại chỗ.

    claim dạng chuỗi -> {"text", "doc_id"}; khoá lạ ("quote", "source"...) ->
    khoá chuẩn; "DOC-4" -> "doc-0004"; abstain dạng chuỗi -> bool. Chữ của
    claim giữ nguyên từng ký tự: đây chỉ là đổi vỏ, không đổi nội dung.
    """
    if not isinstance(report, dict):
        return
    abstain = report.get("abstain")
    if isinstance(abstain, str):
        report["abstain"] = norm(abstain) in ("true", "yes", "1", "có", "đúng")
    elif isinstance(abstain, (int, float)) and not isinstance(abstain, bool):
        report["abstain"] = bool(abstain)

    claims = report.get("claims")
    if isinstance(claims, (dict, str)):
        claims = [claims]
    if not isinstance(claims, list):
        return
    corpus = getattr(ctx, "corpus", None)
    fixed = []
    for claim in claims:
        if isinstance(claim, str):
            claim = {"text": claim, "doc_id": ""}
        if not isinstance(claim, dict):
            fixed.append(claim)
            continue
        if not isinstance(claim.get("text"), str):
            for key in _TEXT_KEYS:
                if isinstance(claim.get(key), str):
                    claim["text"] = claim[key]
                    break
        doc_id = claim.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            for key in _DOC_KEYS:
                value = claim.get(key)
                if isinstance(value, list) and value and isinstance(value[0], str):
                    value = value[0]
                if isinstance(value, str) and value.strip():
                    doc_id = value
                    break
        if isinstance(doc_id, str):
            claim["doc_id"] = canonical_doc_id(doc_id.strip(), corpus)
        fixed.append(claim)
    report["claims"] = fixed
