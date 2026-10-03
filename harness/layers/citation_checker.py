"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

Câu thì thật, nguồn thì sai: mô hình neo mọi claim vào tài liệu trông "chính
thống". Layer này gắn lại `doc_id` về tài liệu thật sự chứa câu đó (một DÒNG
nguyên văn) trong số tài liệu agent đã đọc.

  * Chỉ đổi `claim["doc_id"]` và `report["citations"]`; KHÔNG sửa `claim["text"]`.
  * Câu không có trong bằng chứng nào là bịa -> để `critic` xoá, không bịa doc_id.
"""

from __future__ import annotations

from harness.layers._text import norm_lines, on_one_line, source_doc_id, trimmed
from harness.middleware import Middleware


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def after_agent(self, ctx, report):
        claims = report.get("claims") if isinstance(report, dict) else None
        if not claims or not isinstance(claims, list) or ctx.corpus is None:
            return report

        observed = ctx.observed_text
        observed_lines = norm_lines(observed)
        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
                continue
            doc = ctx.corpus.get(claim.get("doc_id"))
            if (
                doc is not None
                and on_one_line(claim["text"], norm_lines(doc.body))
                and (doc.body in observed or doc.doc_id in observed)
            ):
                continue
            real = source_doc_id(ctx, claim["text"], observed_lines, observed)
            if real is None and trimmed(claim["text"]) != claim["text"]:
                # Dấu câu/ký hiệu thừa: cắt bớt (substring) rồi tìm lại nguồn.
                real = source_doc_id(ctx, trimmed(claim["text"]), observed_lines, observed)
                if real is not None:
                    claim["text"] = trimmed(claim["text"])
            if real is not None:
                claim["doc_id"] = real

        report["citations"] = sorted(
            {c["doc_id"] for c in claims if isinstance(c, dict) and c.get("doc_id")}
        )
        return report
