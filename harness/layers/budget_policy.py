"""LỚP `budget_policy` — bài giảng Day 16, §3 (Budgets & Control Flow).

Kế hoạch của mô hình luôn dài 11 lượt công cụ, 4 lượt cuối là rác. Khi ngân
sách chỉ còn phần dự trữ cho `submit`:
  * `before_model` nhắc mô hình chốt FINAL (câu nhắc PHẢI chứa
    `FINALIZE_SENTINEL`, và trả về danh sách MỚI, không append);
  * `wrap_tool_call` từ chối gọi thêm công cụ (không raise).
Không nén ngữ cảnh ở đây: mô hình chỉ trích được câu còn nguyên văn.
"""

from __future__ import annotations

from arena.model import FINALIZE_SENTINEL
from arena.tools import ToolResult

from harness.middleware import Middleware

#: Dành lại cho lượt `submit` mà agent vẫn còn phải gọi.
DEFAULT_RESERVE = 1

NUDGE = (
    "Ngân sách công cụ đã hết. Hãy trả lời ngay bằng bằng chứng đang có, "
    f"không gọi thêm công cụ nào nữa. {FINALIZE_SENTINEL}"
)


class BudgetPolicy(Middleware):
    """Ép mô hình chốt FINAL ngay khi ngân sách công cụ đã tiêu hết."""

    name = "budget_policy"

    def __init__(self, reserve: int = DEFAULT_RESERVE) -> None:
        self.reserve = max(0, int(reserve))

    def _spent(self, ctx) -> bool:
        limit = ctx.max_tool_calls
        if limit is None:
            return False
        return ctx.tools.calls >= limit - self.reserve

    def before_model(self, ctx, messages):
        if not self._spent(ctx):
            return messages
        return messages + [{"role": "user", "content": NUDGE}]

    def wrap_tool_call(self, ctx, call, name, args):
        if not self._spent(ctx):
            return call(name, args)
        return ToolResult(ok=False, content="", error="Hết ngân sách công cụ")
