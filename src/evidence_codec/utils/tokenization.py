from __future__ import annotations


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count final-LLM-style tokens with tiktoken when available."""

    try:
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        return len(text.split())


def truncate_by_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    if max_tokens <= 0:
        return ""
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
        token_ids = encoding.encode(text)
        return encoding.decode(token_ids[:max_tokens])
    except Exception:
        return " ".join(text.split()[:max_tokens])


def token_windows(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    if max_tokens <= 0:
        return []
    step = max_tokens - max(0, overlap_tokens)
    if step <= 0:
        step = max_tokens
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
        token_ids = encoding.encode(text)
        windows = []
        for start in range(0, len(token_ids), step):
            window_ids = token_ids[start : start + max_tokens]
            if not window_ids:
                break
            windows.append(encoding.decode(window_ids))
        return windows
    except Exception:
        words = text.split()
        return [" ".join(words[start : start + max_tokens]) for start in range(0, len(words), step)]
