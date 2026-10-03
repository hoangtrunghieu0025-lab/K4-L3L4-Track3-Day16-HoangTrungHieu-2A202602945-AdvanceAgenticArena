"""LỚP `critic` — bài giảng Day 16, §2 (Reflection & Self-Critique).

Mô hình không bao giờ nói "không biết" — nó bịa. Tín hiệu giống hệt scorer: claim
có phải là trích dẫn nguyên văn MỘT dòng của một tài liệu ĐÃ TRUY XUẤT không.

Với mỗi claim (sau khi `citation_checker` đã gắn lại nguồn):
  * Đúng là trích dẫn -> giữ nguyên.
  * Chỉ lệch vì dấu câu/ký hiệu bọc ngoài -> cắt bớt (substring) rồi giữ.
  * Là câu GHÉP từ nhiều đoạn trích (nối bằng "và", ";", "…", ", "...) -> tách
    tại chỗ nối, giữ những đoạn là trích dẫn thật (mỗi đoạn vẫn là chữ của mô
    hình), miễn chúng phủ phần lớn câu gốc.
  * Còn lại là bịa / diễn đạt lại -> xoá (một claim HALLUCINATED mất trọn 15
    điểm honesty).

Rồi quyết định `abstain` từ NỘI DUNG, không từ nhãn:
  * Không còn claim nào -> abstain (tránh bị chấm "không có bài nộp").
  * Các claim còn lại chỉ nói "không có / chưa có số liệu" -> abstain.
  * Hai tài liệu CÙNG CHỦ ĐỀ, đều có thẩm quyền, đưa ra CON SỐ KHÁC NHAU cho
    cùng một điều -> mâu thuẫn -> abstain (sau khi đã nêu cả hai phía).
Không bao giờ đổi `abstain` từ True về False.

Chỉ xoá / giữ / cắt bớt — không bao giờ viết lại chữ của claim.
"""

from __future__ import annotations

import re

from harness.layers._text import (
    MIN_QUOTE_CHARS,
    content_words,
    family,
    non_authoritative,
    norm,
    norm_lines,
    normalise_report,
    numbers,
    retrieved_ids,
    says_insufficient,
    trimmed,
)
from harness.middleware import Middleware

#: Một đoạn tách ra phải đủ dài mới đáng tin là trích dẫn chứ không phải mảnh vụn.
_MIN_PIECE_CHARS = 20
#: Các đoạn giữ lại phải phủ ít nhất ngần này độ dài câu gốc, nếu không thì
#: câu gốc chủ yếu là chữ bịa và một mảnh trùng hợp không cứu được nó.
_MIN_COVERAGE = 0.6

#: Giới hạn của scorer: tối đa claim/tài liệu (REDUNDANT), độ dài (OVERLONG).
_MAX_PER_DOC = 4
_MAX_CLAIM_CHARS = 480
#: Tổng số claim giữ lại; dư thì bỏ claim ít liên quan câu hỏi nhất
#: (claim không phủ dữ kiện nào quá hạn mức bị chấm IRRELEVANT).
_MAX_CLAIMS = 6

#: Ngưỡng giống nhau (Jaccard từ nội dung) để hai câu được coi là nói về cùng
#: một điều khi so con số.
_CONFLICT_OVERLAP = 0.12

_DATE = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
_CODE = re.compile(r"\b[a-z]{1,5}-\d+\b")

ABSTAIN_NOTE = "Không đủ căn cứ trong tài liệu đã truy xuất để trả lời chắc chắn."


def _quantities(text) -> frozenset:
    """Con số "định lượng" của câu (bỏ ngày tháng, mã văn bản, năm)."""
    cleaned = _CODE.sub(" ", _DATE.sub(" ", norm(text)))
    return frozenset(n for n in numbers(cleaned) if n.isdigit() and len(n) <= 3)


class Critic(Middleware):
    """Xoá những gì bằng chứng không đỡ; abstain khi không còn gì."""

    name = "critic"

    # -- tín hiệu -------------------------------------------------------

    @staticmethod
    def _index(ctx, retrieved):
        """[(doc_id, các dòng chuẩn hoá)] của tài liệu đã truy xuất, tài liệu
        có toàn văn trong quan sát đứng trước."""
        if ctx.corpus is None:
            return [(None, norm_lines(ctx.observed_text))]
        observed = ctx.observed_text
        docs = [d for d in ctx.corpus.docs if d.doc_id in retrieved]
        docs.sort(key=lambda d: 0 if d.body in observed else 1)
        return [(d.doc_id, norm_lines(d.body)) for d in docs]

    @staticmethod
    def _find(index, text, prefer):
        """doc_id có một dòng chứa nguyên văn `text` (ưu tiên `prefer`), hoặc None."""
        needle = norm(text)
        if len(needle) < MIN_QUOTE_CHARS:
            return None
        hits = [doc_id for doc_id, lines in index if any(needle in line for line in lines)]
        if not hits:
            return None
        if prefer in hits:
            return prefer
        return hits[0] if hits[0] is not None else (prefer or "")

    def _pieces(self, index, text, prefer):
        """Tách câu ghép tại ranh giới từ thành các đoạn trích thật.

        Tham lam từ trái sang: ở mỗi vị trí bắt đầu, lấy đoạn DÀI NHẤT là trích
        dẫn nguyên văn; không có thì bỏ qua một từ. Mỗi đoạn là substring của
        chữ mô hình đã viết, nên provenance còn nguyên.
        """
        gaps = list(re.finditer(r"\s+", text))
        starts = [0] + [g.end() for g in gaps]
        ends = [g.start() for g in gaps] + [len(text)]
        pieces, i = [], 0
        while i < len(starts):
            found = None
            for j in range(len(ends) - 1, i - 1, -1):
                piece = trimmed(text[starts[i]:ends[j]])
                if len(piece) < _MIN_PIECE_CHARS:
                    break
                source = self._find(index, piece, prefer)
                if source is not None:
                    found = (j, piece, source)
                    break
            if found:
                pieces.append(found[1:])
                i = found[0] + 1
            else:
                i += 1
        covered = sum(len(piece) for piece, _ in pieces)
        whole = len(trimmed(text))
        return pieces if pieces and covered >= _MIN_COVERAGE * whole else []

    def _conflict(self, ctx, claims) -> bool:
        """Hai nguồn có thẩm quyền, cùng chủ đề, cho con số khác nhau."""
        if ctx.corpus is None:
            return False
        for i, a in enumerate(claims):
            for b in claims[i + 1:]:
                if a.get("doc_id") == b.get("doc_id"):
                    continue
                doc_a, doc_b = ctx.corpus.get(a.get("doc_id")), ctx.corpus.get(b.get("doc_id"))
                if doc_a is None or doc_b is None or not family(doc_a):
                    continue
                if family(doc_a) != family(doc_b):
                    continue
                if non_authoritative(doc_a) or non_authoritative(doc_b):
                    continue
                qa, qb = _quantities(a["text"]), _quantities(b["text"])
                if not qa or not qb or qa == qb:
                    continue
                wa, wb = content_words(a["text"]), content_words(b["text"])
                if wa and wb and len(wa & wb) / len(wa | wb) >= _CONFLICT_OVERLAP:
                    return True
        return False

    # -- hook -----------------------------------------------------------

    def after_agent(self, ctx, report):
        if not isinstance(report, dict):
            return report
        normalise_report(ctx, report)
        claims = report.get("claims")
        claims = claims if isinstance(claims, list) else []
        index = self._index(ctx, retrieved_ids(ctx))

        kept = []
        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
                continue  # MALFORMED: bỏ còn hơn bị phạt
            text = claim["text"]
            if len(text) > _MAX_CLAIM_CHARS:
                text = claim["text"] = text[:_MAX_CLAIM_CHARS]
            prefer = claim.get("doc_id")
            source = self._find(index, text, prefer)
            if source is None and trimmed(text) != text:
                source = self._find(index, trimmed(text), prefer)
                if source is not None:
                    text = claim["text"] = trimmed(text)
            if source is not None:
                # Có trong bằng chứng -> không phải bịa. Gắn đúng nguồn là việc
                # của `citation_checker` (đã chạy trước); critic không đụng doc_id.
                kept.append(claim)
                continue
            kept.extend(
                {"text": piece, "doc_id": doc_id}
                for piece, doc_id in self._pieces(index, text, prefer)
            )

        kept = self._limit(ctx, self._dedupe(kept))
        report["claims"] = kept
        report["citations"] = sorted({c["doc_id"] for c in kept if c.get("doc_id")})

        answer = report.get("answer")
        answer = answer if isinstance(answer, str) else ""
        if not kept:
            report["abstain"] = True
            report["citations"] = []
            report["answer"] = f"{ABSTAIN_NOTE} {answer}".strip()
        elif all(says_insufficient(c["text"]) for c in kept) or (
            says_insufficient(answer) and any(says_insufficient(c["text"]) for c in kept)
        ):
            report["abstain"] = True
        elif self._conflict(ctx, kept):
            report["abstain"] = True
        return report

    # -- dọn dẹp ----------------------------------------------------------

    @staticmethod
    def _dedupe(claims):
        """Bỏ claim trùng, hoặc nằm gọn trong một claim khác của cùng tài liệu."""
        result = []
        for claim in claims:
            text = norm(claim["text"])
            if any(
                text == norm(other["text"])
                or (text in norm(other["text"]) and claim.get("doc_id") == other.get("doc_id"))
                for other in result
            ):
                continue
            result = [
                other
                for other in result
                if not (norm(other["text"]) in text and claim.get("doc_id") == other.get("doc_id"))
            ]
            result.append(claim)
        return result

    @staticmethod
    def _limit(ctx, claims):
        per_doc, capped = {}, []
        for claim in claims:
            doc_id = claim.get("doc_id")
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            if per_doc[doc_id] <= _MAX_PER_DOC:
                capped.append(claim)
        if len(capped) <= _MAX_CLAIMS:
            return capped
        asked = content_words(ctx.brief.get("question_vi", "") if isinstance(ctx.brief, dict) else "")
        ranked = sorted(
            range(len(capped)),
            key=lambda i: (-len(content_words(capped[i]["text"]) & asked), i),
        )
        keep = set(ranked[:_MAX_CLAIMS])
        return [claim for i, claim in enumerate(capped) if i in keep]
