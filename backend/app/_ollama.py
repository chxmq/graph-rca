from __future__ import annotations


def response_text(response) -> str:
    if response is None:
        return ""
    if hasattr(response, "response"):
        return str(response.response or "")
    if isinstance(response, dict):
        return str(response.get("response", ""))
    return str(response)
