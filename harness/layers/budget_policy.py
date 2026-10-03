"""LỚP `budget_policy` — bài giảng Day 16, §3 (Budgets & Control Flow).

Kế hoạch của mô hình luôn dài 11 lượt công cụ, 4 lượt cuối là rác. Lớp này:
  * `before_model`  — khi ngân sách công cụ chỉ còn phần dự trữ cho `submit`
    (hoặc token sắp vượt xa ngân sách), nhắc mô hình chốt FINAL. Câu nhắc PHẢI
    chứa `FINALIZE_SENTINEL`, và trả về danh sách MỚI (không append).
    Đồng thời rút gọn các kết quả search ĐÃ CŨ (chỉ giữ doc_id + tiêu đề): mỗi
    lượt gọi mô hình gửi lại toàn bộ lịch sử, nên đó là token bị trả nhiều lần.
    Toàn văn tài liệu đã fetch thì KHÔNG đụng tới — mô hình cần trích từ đó.
  * `wrap_model_call` — đếm token đã tiêu.
  * `wrap_tool_call` — từ chối gọi công cụ khi đã cạn (không raise), và không
    tốn lượt cho một lời gọi y hệt lời gọi đã thành công trước đó.
"""

from __future__ import annotations

import json

from arena.model import FINALIZE_SENTINEL, is_degraded
from arena.tools import ToolResult

from harness.agent import _as_k, _as_text
from harness.layers._text import canonical_doc_id, max_tokens, max_tool_calls, norm
from harness.middleware import Middleware

#: Dành lại cho lượt `submit` mà agent vẫn còn phải gọi.
DEFAULT_RESERVE = 1

#: Chốt sớm nếu (token đã tiêu + hai lượt nữa) vượt ngần này lần ngân sách
#: token — tức là sắp rơi vào bậc 0 điểm token của scorer. Đặt rộng tay có chủ
#: ý: grounding (55 điểm) đáng hơn token (6 điểm), không cắt lúc còn đang tìm.
TOKEN_SLACK = 1.75
#: Ước lượng phần lịch sử tăng thêm mỗi lượt (quan sát + output).
_STEP_GROWTH = 400

NUDGE = (
    "Ngân sách công cụ đã hết. Hãy trả lời ngay bằng bằng chứng đang có, "
    "không gọi thêm công cụ nào nữa: viết THOUGHT rồi dòng FINAL với các câu "
    f"trích nguyên văn từ tài liệu đã đọc. {FINALIZE_SENTINEL}"
)
BLOCKED = (
    "Hết ngân sách công cụ — lời gọi này KHÔNG được thực hiện. "
    "Không gọi thêm công cụ; hãy trả lời ngay bằng dòng FINAL."
)
DUPLICATE = (
    "Lời gọi này trùng hệt một lời gọi đã thành công ở lượt trước — kết quả đã có "
    "ở phía trên, không gọi lại. Hãy dùng nội dung đó hoặc tìm với truy vấn khác."
)
_SEARCH_DIGEST = "[Kết quả search cũ, đã rút gọn: doc_id | tiêu đề]"


class BudgetPolicy(Middleware):
    """Ép mô hình chốt FINAL ngay khi ngân sách công cụ đã tiêu hết."""

    name = "budget_policy"

    def __init__(self, reserve: int = DEFAULT_RESERVE, token_slack: float = TOKEN_SLACK) -> None:
        self.reserve = max(0, int(reserve))
        self.token_slack = float(token_slack)

    # -- tín hiệu -------------------------------------------------------

    def _calls_spent(self, ctx) -> bool:
        return ctx.tools.calls >= max_tool_calls(ctx) - self.reserve

    def _tokens_spent(self, ctx) -> bool:
        used = ctx.state.get("tokens_used", 0)
        last = ctx.state.get("last_call_tokens", 0)
        if not last:
            return False
        return used + 2 * (last + _STEP_GROWTH) > self.token_slack * max_tokens(ctx)

    def _spent(self, ctx) -> bool:
        return self._calls_spent(ctx) or self._tokens_spent(ctx)

    # -- hook -----------------------------------------------------------

    def before_model(self, ctx, messages):
        messages = self._compact(messages)
        if not self._spent(ctx):
            return messages
        return messages + [{"role": "user", "content": NUDGE}]

    def wrap_model_call(self, ctx, call, messages):
        response = call(messages)
        spent = 0
        for field in ("prompt_tokens", "completion_tokens"):
            value = getattr(response, field, 0)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                spent += value
        ctx.state["tokens_used"] = ctx.state.get("tokens_used", 0) + spent
        ctx.state["last_call_tokens"] = spent
        return response

    def wrap_tool_call(self, ctx, call, name, args):
        if self._spent(ctx):
            return ToolResult(ok=False, content="", error=BLOCKED)
        key = self._key(ctx, name, args)
        done = ctx.state.setdefault("done_calls", set())
        if key is not None and key in done:
            return ToolResult(ok=True, content=DUPLICATE, error=None)
        result = call(name, args)
        content = result.content if isinstance(result.content, str) else ""
        if key is not None and result.ok and content and not is_degraded(content):
            done.add(key)
        return result

    # -- phụ trợ ----------------------------------------------------------

    @staticmethod
    def _key(ctx, name, args):
        args = args if isinstance(args, dict) else {}
        if name == "fetch_doc":
            return ("fetch_doc", canonical_doc_id(_as_text(args.get("doc_id")).strip(), ctx.corpus))
        if name == "search":
            return ("search", norm(_as_text(args.get("query"))), _as_k(args.get("k")))
        return None

    @staticmethod
    def _compact(messages):
        """Rút gọn các kết quả search không phải quan sát MỚI NHẤT."""
        last_user = max(
            (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"),
            default=-1,
        )
        out = []
        for i, message in enumerate(messages):
            digest = None
            if i != last_user and i > 1 and isinstance(message, dict) and message.get("role") == "user":
                digest = _digest_search(message.get("content"))
            out.append({**message, "content": digest} if digest else message)
        return out


def _digest_search(content):
    """JSON kết quả search -> danh sách "doc_id | tiêu đề"; None nếu không phải."""
    if not isinstance(content, str) or not content.startswith("[{") or '"snippet"' not in content:
        return None
    payload, _, hint = content.partition("\n[")
    try:
        rows = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(rows, list) or not all(isinstance(r, dict) and "doc_id" in r for r in rows):
        return None
    lines = [f"- {r.get('doc_id')} | {r.get('title', '')}" for r in rows]
    digest = _SEARCH_DIGEST + "\n" + "\n".join(lines)
    return digest + ("\n[" + hint if hint else "")
