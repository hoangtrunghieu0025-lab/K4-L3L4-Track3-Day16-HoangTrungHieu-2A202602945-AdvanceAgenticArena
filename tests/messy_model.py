"""Mô hình giả "lộn xộn": bọc MockModel rồi viết lại dòng FINAL theo kiểu mô hình thật.

Mục đích: kiểm tra layer có trụ được khi đầu ra KHÔNG đúng một hình dạng cố định.
Mọi biến đổi đều là thứ một mô hình thật có thể viết ra; chữ được ghi vào trace
chính là chữ đã biến đổi, nên scorer chấm provenance trên đúng đầu ra này.
"""

from __future__ import annotations

import json
import re

from arena.model import BaseModel, MockModel, ModelResponse

MUTATIONS = (
    "clean",       # nguyên bản mô hình giả
    "period",      # thêm dấu chấm cuối claim
    "bullet",      # "- " đầu claim
    "bold",        # **claim**
    "fence",       # thụt lề + bọc ``` quanh cả lượt
    "paraphrase",  # thêm một claim diễn đạt lại (bịa)
    "lower",       # viết thường
    "tail",        # claim + đuôi tự viết ", theo quy định hiện hành"
    "strings",     # claims là list chuỗi, không có doc_id
    "keys",        # {"quote": ..., "source": "DOC-4"} thay vì {"text", "doc_id"}
    "abstain_str", # "abstain": "false" (chuỗi)
    "all_noise",   # period + bullet + bold + fence + paraphrase
)

_PARAPHRASE = "Theo ghi nhận chung, các bước xử lý thường hoàn tất trong vài ngày làm việc tới."


def _mutate_text(text: str, mode: str) -> str:
    if mode in ("period", "all_noise"):
        text = text + "."
    if mode in ("bullet", "all_noise"):
        text = "- " + text
    if mode in ("bold", "all_noise"):
        text = f"**{text}**"
    if mode == "lower":
        text = text.lower()
    if mode == "tail":
        text = text + ", theo quy định hiện hành"
    return text


def _short_id(doc_id: str) -> str:
    match = re.fullmatch(r"doc-0*(\d+)", doc_id or "")
    return f"DOC-{match.group(1)}" if match else doc_id


class MessyModel(BaseModel):
    def __init__(self, corpus, seed: int, mode: str) -> None:
        self._inner = MockModel(corpus=corpus, seed=seed)
        self.mode = mode

    def complete(self, messages, **kw) -> ModelResponse:
        response = self._inner.complete(messages, **kw)
        if self.mode == "clean" or "FINAL:" not in response.text:
            return response
        head, _, tail = response.text.partition("FINAL:")
        try:
            payload = json.loads(tail.strip())
        except ValueError:
            return response
        claims = payload.get("claims")
        if isinstance(claims, list):
            for claim in claims:
                if isinstance(claim, dict) and isinstance(claim.get("text"), str):
                    claim["text"] = _mutate_text(claim["text"], self.mode)
            if self.mode in ("paraphrase", "all_noise") and claims:
                claims.append({"text": _PARAPHRASE, "doc_id": claims[0].get("doc_id", "doc-0001")})
            if self.mode == "strings":
                payload["claims"] = [c["text"] for c in claims if isinstance(c, dict)]
            if self.mode == "keys":
                payload["claims"] = [
                    {"quote": c.get("text"), "source": _short_id(c.get("doc_id", ""))}
                    for c in claims
                    if isinstance(c, dict)
                ]
        if self.mode == "abstain_str":
            payload["abstain"] = "true" if payload.get("abstain") is True else "false"
        payload["answer"] = str(payload.get("answer", "")) + " Hy vọng thông tin này hữu ích."
        out = head + "FINAL: " + json.dumps(payload, ensure_ascii=False)
        if self.mode in ("fence", "all_noise"):
            out = "```\n" + "\n".join("  " + line for line in out.splitlines()) + "\n```"
        return ModelResponse(
            text=out,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
