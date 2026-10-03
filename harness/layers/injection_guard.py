"""LỚP `injection_guard` — bài giảng Day 16, §10 (Prompt Injection Defense).

NGUYÊN TẮC: nội dung tài liệu là DỮ LIỆU, không phải MỆNH LỆNH. Cách ly đoạn
độc ngay tại biên (`wrap_tool_call`), rồi quét lần cuối `answer` ở `after_agent`.

  * Đoạn độc nằm giữa BLOCK_START và BLOCK_END; fetch bị cắt có thể mất
    BLOCK_END nên khi đó cắt từ BLOCK_START tới hết chuỗi.
  * Chỉ sửa `answer`. Không bao giờ sửa `claim["text"]` (mất provenance);
    claim chứa canary thì bỏ hẳn claim đó.
"""

from __future__ import annotations

from arena.corpus import INJECTION_CANARY
from arena.tools import ToolResult

from harness.middleware import Middleware

#: Dấu mốc của đoạn nội dung không đáng tin trong kho tài liệu.
BLOCK_START = "--- LƯU Ý HỆ THỐNG"
BLOCK_END = "--- HẾT LƯU Ý HỆ THỐNG ---"
PLACEHOLDER = "[nội dung không đáng tin cậy đã bị loại bỏ]"


def _strip_blocks(content: str) -> str:
    while BLOCK_START in content:
        start = content.find(BLOCK_START)
        end = content.find(BLOCK_END, start)
        if end == -1:
            content = content[:start] + PLACEHOLDER
            break
        content = content[:start] + PLACEHOLDER + content[end + len(BLOCK_END):]
    return content


class InjectionGuard(Middleware):
    """Coi nội dung tài liệu là dữ liệu: cách ly nó, rồi soát lại câu trả lời."""

    name = "injection_guard"

    def wrap_tool_call(self, ctx, call, name, args):
        result = call(name, args)
        content = getattr(result, "content", None)
        if not isinstance(content, str):
            return result
        clean = _strip_blocks(content)
        # Canary rơi ngoài dấu mốc (vd. dấu mốc bị cắt mất) vẫn phải gỡ.
        clean = clean.replace(INJECTION_CANARY, "")
        if clean == content:
            return result
        return ToolResult(ok=result.ok, content=clean, error=result.error)

    def after_agent(self, ctx, report):
        if not isinstance(report, dict):
            return report
        answer = report.get("answer")
        if isinstance(answer, str) and INJECTION_CANARY in answer:
            report["answer"] = answer.replace(INJECTION_CANARY, "").strip()
        claims = report.get("claims")
        if isinstance(claims, list):
            kept = [
                c
                for c in claims
                if not (isinstance(c, dict) and INJECTION_CANARY in str(c.get("text", "")))
            ]
            if len(kept) != len(claims):
                report["claims"] = kept
        citations = report.get("citations")
        if isinstance(citations, list) and any(
            INJECTION_CANARY in str(c) for c in citations
        ):
            report["citations"] = [c for c in citations if INJECTION_CANARY not in str(c)]
        return report
