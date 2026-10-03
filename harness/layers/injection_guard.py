"""LỚP `injection_guard` — bài giảng Day 16, §10 (Prompt Injection Defense).

NGUYÊN TẮC: nội dung tài liệu là DỮ LIỆU, không phải MỆNH LỆNH. Cách ly đoạn
độc ngay tại biên (`wrap_tool_call`), rồi quét lần cuối báo cáo (`after_agent`).

  * Đoạn độc nằm giữa BLOCK_START và BLOCK_END; fetch bị cắt có thể mất
    BLOCK_END nên khi đó cắt từ BLOCK_START tới hết chuỗi.
  * Scorer tìm canary trong TOÀN BỘ báo cáo (mọi trường), nên quét hết.
  * Không bao giờ sửa `claim["text"]` (mất provenance); claim chứa canary thì
    bỏ hẳn claim đó.
"""

from __future__ import annotations

import re

from arena.corpus import INJECTION_CANARY
from arena.tools import ToolResult

from harness.middleware import Middleware

#: Dấu mốc của đoạn nội dung không đáng tin trong kho tài liệu.
BLOCK_START = "--- LƯU Ý HỆ THỐNG"
BLOCK_END = "--- HẾT LƯU Ý HỆ THỐNG ---"
PLACEHOLDER = "[nội dung không đáng tin cậy đã bị loại bỏ]"

_CANARY = re.compile(re.escape(INJECTION_CANARY), re.IGNORECASE)


def _strip_blocks(content: str) -> str:
    while BLOCK_START in content:
        start = content.find(BLOCK_START)
        end = content.find(BLOCK_END, start)
        if end == -1:
            content = content[:start] + PLACEHOLDER
            break
        content = content[:start] + PLACEHOLDER + content[end + len(BLOCK_END):]
    return content


def _scrub(value):
    """Gỡ canary khỏi mọi chuỗi trong một cấu trúc JSON-like."""
    if isinstance(value, str):
        return _CANARY.sub("", value)
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, dict):
        return {_scrub(k) if isinstance(k, str) else k: _scrub(v) for k, v in value.items()}
    return value


class InjectionGuard(Middleware):
    """Coi nội dung tài liệu là dữ liệu: cách ly nó, rồi soát lại câu trả lời."""

    name = "injection_guard"

    def wrap_tool_call(self, ctx, call, name, args):
        result = call(name, args)
        content = getattr(result, "content", None)
        if not isinstance(content, str):
            return result
        # Canary rơi ngoài dấu mốc (vd. dấu mốc bị cắt mất) vẫn phải gỡ.
        clean = _CANARY.sub("", _strip_blocks(content))
        if clean == content:
            return result
        return ToolResult(ok=result.ok, content=clean, error=result.error)

    def after_agent(self, ctx, report):
        if not isinstance(report, dict):
            return report
        claims = report.get("claims")
        if isinstance(claims, list):
            report["claims"] = [claim for claim in claims if not _CANARY.search(str(claim))]
        for key in list(report):
            if key != "claims":
                report[key] = _scrub(report[key])
        if isinstance(report.get("answer"), str):
            report["answer"] = report["answer"].strip()
        return report
