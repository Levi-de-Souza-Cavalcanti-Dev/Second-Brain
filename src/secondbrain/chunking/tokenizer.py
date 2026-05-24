"""Token counting utilities for chunking."""

from __future__ import annotations

from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


class CharTokenCounter:
    """Fallback: 1 char ≈ 1 token."""

    def count(self, text: str) -> int:
        return len(text)

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")


class TiktokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)


class HFTokenCounter:
    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        result = self._tok.decode(tokens, skip_special_tokens=True)
        return str(result)


def make_token_counter(*, by_tokens: bool, model_hint: str = "") -> TokenCounter:
    if not by_tokens:
        return CharTokenCounter()
    try:
        return TiktokenCounter()
    except ImportError:
        pass
    if model_hint:
        try:
            return HFTokenCounter(model_hint)
        except Exception:
            pass
    return CharTokenCounter()
