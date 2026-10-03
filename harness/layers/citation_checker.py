"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

Câu thì thật, nguồn thì sai: mô hình neo mọi claim vào tài liệu trông "chính
thống". Layer này gắn lại `doc_id` về tài liệu ĐÃ TRUY XUẤT thật sự chứa câu đó
(nguyên văn, trong MỘT dòng).

Ba việc:
  1. `wrap_tool_call` — ghi lại tập tài liệu mà scorer sẽ coi là "đã truy xuất"
     (mỗi fetch_doc + phát lại đúng mỗi search), để không bao giờ gắn claim vào
     tài liệu chưa đọc (`UNRETRIEVED`).
  2. `wrap_tool_call` — sau mỗi search, liệt kê thêm các tài liệu CÙNG CHỦ ĐỀ
     (cùng tiền tố tiêu đề) với top kết quả mà search chưa trả về. Văn bản chính
     thức chứa câu trả lời thường nằm ngoài top-k, cạnh bản Hỏi & Đáp/Báo cáo
     cùng chủ đề. Chỉ là gợi ý (id + tiêu đề) — mô hình vẫn tự quyết fetch gì.
  3. `after_agent` — chuẩn hoá HÌNH DẠNG báo cáo, rồi gắn lại doc_id.

ĐƯỢC: đổi `claim["doc_id"]`, cắt bớt dấu câu bọc ngoài claim (substring).
KHÔNG: viết lại chữ của claim. Câu không có ở tài liệu đã truy xuất nào thì để
`critic` xử lý — không bịa doc_id.
"""

from __future__ import annotations

from arena.model import is_degraded
from arena.tools import ToolResult

from harness.agent import _as_k, _as_text
from harness.layers._text import (
    canonical_doc_id,
    family,
    normalise_report,
    norm_lines,
    on_one_line,
    retrieved_ids,
    source_doc_id,
    trimmed,
)
from harness.middleware import Middleware

#: Gợi ý tài liệu cùng chủ đề sau mỗi search (đặt False để tắt).
SUGGEST_SIBLINGS = True
#: Gợi ý tối đa bao nhiêu tài liệu sau một lần search.
MAX_SIBLINGS = 6
SIBLING_HEADER = (
    "[Gợi ý từ hệ thống: tài liệu CÙNG CHỦ ĐỀ với kết quả trên nhưng chưa có "
    "trong danh sách — dùng fetch_doc nếu cần]"
)


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    # -- trên đường vào: ghi nhận những gì đã truy xuất -------------------

    def wrap_tool_call(self, ctx, call, name, args):
        before = ctx.tools.calls
        result = call(name, args)
        if ctx.tools.calls == before or ctx.corpus is None:
            return result  # công cụ không thật sự chạy (bị chặn / trùng lặp)
        args = args if isinstance(args, dict) else {}
        found = ctx.state.setdefault("retrieved", set())
        if name == "fetch_doc":
            doc_id = canonical_doc_id(_as_text(args.get("doc_id")).strip(), ctx.corpus)
            if ctx.corpus.get(doc_id) is not None:
                found.add(doc_id)
        elif name == "search":
            query = _as_text(args.get("query"))
            hits = ctx.corpus.search(query, k=_as_k(args.get("k"))) if query else []
            found.update(d.doc_id for d in hits)
            content = result.content if isinstance(result.content, str) else ""
            if SUGGEST_SIBLINGS and hits and result.ok and not is_degraded(content):
                hint = self._siblings(ctx, hits)
                if hint:
                    return ToolResult(ok=result.ok, content=content + "\n" + hint, error=result.error)
        return result

    def _siblings(self, ctx, hits) -> str:
        """Mỗi chủ đề trong kết quả (theo thứ hạng) gợi ý trước MỘT tài liệu
        chưa hiện — ưu tiên văn bản chính thức — rồi mới tới tài liệu thứ hai."""
        shown = {d.doc_id for d in hits}
        queues = []
        for topic in dict.fromkeys(family(d) for d in hits if family(d)):
            siblings = [d for d in ctx.corpus.docs if d.doc_id not in shown and family(d) == topic]
            siblings.sort(key=lambda d: "chính thức" not in d.title.casefold())
            if siblings:
                queues.append(siblings)
        picked = []
        while queues and len(picked) < MAX_SIBLINGS:
            for queue in list(queues):
                if len(picked) >= MAX_SIBLINGS:
                    break
                picked.append(queue.pop(0))
                if not queue:
                    queues.remove(queue)
        lines = [f"- {d.doc_id} | {d.title}" for d in picked]
        return SIBLING_HEADER + "\n" + "\n".join(lines) if lines else ""

    # -- trên đường ra: gắn lại nguồn ---------------------------------------

    def after_agent(self, ctx, report):
        if not isinstance(report, dict):
            return report
        normalise_report(ctx, report)
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims or ctx.corpus is None:
            return report

        retrieved = retrieved_ids(ctx)
        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
                continue
            cited = claim.get("doc_id")
            doc = ctx.corpus.get(cited) if isinstance(cited, str) else None
            if (
                doc is not None
                and cited in retrieved
                and on_one_line(claim["text"], norm_lines(doc.body))
            ):
                continue  # đã đúng nguồn
            real = source_doc_id(ctx, claim["text"], prefer=cited, retrieved=retrieved)
            if real is None:
                cut = trimmed(claim["text"])
                if cut != claim["text"]:
                    real = source_doc_id(ctx, cut, prefer=cited, retrieved=retrieved)
                    if real is not None:
                        claim["text"] = cut  # cắt bớt dấu câu thừa: vẫn là substring
            if real is not None:
                claim["doc_id"] = real

        report["citations"] = sorted(
            {
                c["doc_id"]
                for c in claims
                if isinstance(c, dict) and isinstance(c.get("doc_id"), str) and c["doc_id"]
            }
        )
        return report
