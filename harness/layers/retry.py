"""LỚP `retry` — bài giảng Day 16, §7 (Failure Handling & Retries).

Thử lại Ở DƯỚI mô hình (trong `wrap_tool_call`) khi công cụ hỏng TẠM THỜI:
timeout, hoặc `ok=True` nhưng nội dung bị cắt/nhiễu (`is_degraded`). Giá trị
thật là giảm PHƯƠNG SAI.

Không thử lại lỗi VĨNH VIỄN ("doc not found", "invalid expression", tool lạ):
gọi lại y hệt chỉ ra y hệt, mà mỗi lần gọi tốn một lượt ngân sách. Trước khi
gọi, sửa doc_id viết lệch ("DOC-4" -> "doc-0004") để khỏi tốn lượt vì lỗi gõ.
`budget_policy` không nhìn thấy các lượt gọi lại, nên lớp này tự dừng khi chỉ
còn phần dự trữ cho `submit`.
"""

from __future__ import annotations

from arena.model import is_degraded

from harness.layers._text import canonical_doc_id, max_tool_calls
from harness.middleware import Middleware

#: Tổng số lần thử, tính cả lần đầu.
DEFAULT_MAX_ATTEMPTS = 3

#: Số lượt để dành cho `submit` mà agent vẫn còn phải gọi.
DEFAULT_RESERVE = 1

#: Lỗi mà gọi lại y hệt cũng không bao giờ hết.
_PERMANENT = ("doc not found:", "invalid expression:", "unknown tool")


class Retry(Middleware):
    """Gọi lại một lượt công cụ trả về kết quả hỏng hoặc suy giảm."""

    name = "retry"

    def __init__(
        self,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        reserve: int = DEFAULT_RESERVE,
    ) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.reserve = max(0, int(reserve))

    @staticmethod
    def _transient(result) -> bool:
        content = result.content if isinstance(result.content, str) else ""
        error = result.error if isinstance(result.error, str) else ""
        if any(marker in content or marker in error for marker in _PERMANENT):
            return False
        return (not result.ok) or is_degraded(content)

    def _budget_left(self, ctx) -> bool:
        return ctx.tools.calls < max_tool_calls(ctx) - self.reserve

    def wrap_tool_call(self, ctx, call, name, args):
        if name == "fetch_doc" and isinstance(args, dict) and isinstance(args.get("doc_id"), str):
            fixed = canonical_doc_id(args["doc_id"].strip(), ctx.corpus)
            if fixed != args["doc_id"]:
                args = {**args, "doc_id": fixed}
        result = call(name, args)
        attempts = 1
        while attempts < self.max_attempts and self._transient(result) and self._budget_left(ctx):
            result = call(name, args)
            attempts += 1
        if attempts > 1:
            ctx.state["retry_attempts"] = ctx.state.get("retry_attempts", 0) + attempts - 1
        return result
