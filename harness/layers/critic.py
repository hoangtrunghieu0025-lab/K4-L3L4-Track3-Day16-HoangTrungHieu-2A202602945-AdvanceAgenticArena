"""LỚP `critic` — bài giảng Day 16, §2 (Reflection & Self-Critique).

Mô hình không bao giờ nói "không biết" — nó bịa. Tín hiệu: claim có phải là
trích dẫn nguyên văn MỘT dòng của bằng chứng agent đã đọc (và của một tài
liệu trong kho) hay không.

  * Có -> giữ nguyên (không sửa chữ).
  * Không, nhưng tách được thành hai nửa đều là trích dẫn hợp lệ (câu ghép từ
    hai nguồn mâu thuẫn) -> giữ hai nửa, gắn doc_id thật, `abstain = True`.
  * Không tách được -> bịa: xoá claim.
  * Hết claim -> `abstain = True`, nói rõ không đủ căn cứ.
Chỉ xoá / giữ / cắt bớt — không bao giờ viết lại chữ của claim.
"""

from __future__ import annotations

from harness.layers._text import norm_lines, on_one_line, source_doc_id, trimmed
from harness.middleware import Middleware

#: Chỗ có thể mô hình đã dán hai nửa câu vào nhau.
_JOINERS = (" và ", "; ", " — ", " - ", ", ", ". ")

#: Một nửa phải đủ dài mới đáng tin là một trích dẫn chứ không phải mảnh vụn.
_MIN_HALF_CHARS = 25

#: Giới hạn của scorer: tối đa claim/tài liệu, tổng số claim, độ dài một claim.
_MAX_PER_DOC = 4
_MAX_CLAIMS = 10
_MAX_CLAIM_CHARS = 480

ABSTAIN_ANSWER = (
    "Không đủ căn cứ trong tài liệu đã truy xuất để trả lời câu hỏi này "
    "một cách đáng tin cậy."
)


class Critic(Middleware):
    """Xoá những gì bằng chứng không đỡ; abstain khi không còn gì."""

    name = "critic"

    def _grounded(self, ctx, text, observed_lines) -> bool:
        if not on_one_line(text, observed_lines):
            return False
        corpus = ctx.corpus
        return corpus is None or any(on_one_line(text, norm_lines(d.body)) for d in corpus.docs)

    def _split(self, ctx, text, observed_lines, observed):
        """Tách câu ghép thành [(nửa, doc_id), ...] hoặc None."""
        for joiner in _JOINERS:
            start = text.find(joiner)
            while start != -1:
                halves = (text[:start].strip(), text[start + len(joiner):].strip())
                if all(len(h) >= _MIN_HALF_CHARS for h in halves):
                    sources = [source_doc_id(ctx, h, observed_lines, observed) for h in halves]
                    if all(sources) and sources[0] != sources[1]:
                        return list(zip(halves, sources))
                start = text.find(joiner, start + 1)
        return None

    def after_agent(self, ctx, report):
        if not isinstance(report, dict):
            return report
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims:
            return report

        observed = ctx.observed_text
        observed_lines = norm_lines(observed)
        spliced = False
        kept = []
        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
                continue
            text = claim["text"]
            if len(text) > _MAX_CLAIM_CHARS:
                text = claim["text"] = text[:_MAX_CLAIM_CHARS].rstrip()
            if not self._grounded(ctx, text, observed_lines):
                trimmed_text = trimmed(text)
                if trimmed_text != text and self._grounded(ctx, trimmed_text, observed_lines):
                    text = claim["text"] = trimmed_text
            if self._grounded(ctx, text, observed_lines):
                kept.append(claim)
                continue
            parts = self._split(ctx, text, observed_lines, observed)
            if parts:
                spliced = True
                kept.extend({"text": half, "doc_id": doc_id} for half, doc_id in parts)

        per_doc: dict = {}
        limited = []
        for claim in kept:
            doc_id = claim.get("doc_id")
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            if per_doc[doc_id] <= _MAX_PER_DOC and len(limited) < _MAX_CLAIMS:
                limited.append(claim)

        report["claims"] = limited
        report["citations"] = sorted({c["doc_id"] for c in limited if c.get("doc_id")})
        if spliced:
            report["abstain"] = True
        if not limited:
            report["abstain"] = True
            report["citations"] = []
            report["answer"] = ABSTAIN_ANSWER
        return report
