"""LỚP `retry` — bài giảng Day 16, §7 (Failure Handling & Retries).

Thử lại Ở DƯỚI mô hình (trong `wrap_tool_call`) khi công cụ hỏng hoặc trả về
nội dung suy giảm (`ok=True` vẫn có thể bị cắt/nhiễu — dùng `is_degraded`).
Giá trị thật là giảm PHƯƠNG SAI. Mỗi lần gọi lại tốn một lượt ngân sách và
`budget_policy` không nhìn thấy chúng, nên chính lớp này tự dừng khi chỉ còn
phần dự trữ cho `submit`.
"""

from __future__ import annotations

from arena.model import is_degraded

from harness.middleware import Middleware

#: Tổng số lần thử, tính cả lần đầu.
DEFAULT_MAX_ATTEMPTS = 3

#: Số lượt để dành cho `submit` mà agent vẫn còn phải gọi.
DEFAULT_RESERVE = 1


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

    def _broken(self, result) -> bool:
        content = getattr(result, "content", "")
        return (not result.ok) or (isinstance(content, str) and is_degraded(content))

    def _budget_left(self, ctx) -> bool:
        limit = ctx.max_tool_calls
        return limit is None or ctx.tools.calls < limit - self.reserve

    def wrap_tool_call(self, ctx, call, name, args):
        result = call(name, args)
        attempts = 1
        while attempts < self.max_attempts and self._broken(result) and self._budget_left(ctx):
            result = call(name, args)
            attempts += 1
        if attempts > 1:
            ctx.state["retry_attempts"] = ctx.state.get("retry_attempts", 0) + attempts - 1
        return result
