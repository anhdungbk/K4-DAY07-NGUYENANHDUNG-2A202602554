from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Look behind for the terminal punctuation so it remains part of the
        # sentence instead of being discarded by ``re.split``.
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
            if sentence.strip()
        ]
        return [
            " ".join(sentences[index : index + self.max_sentences_per_chunk])
            for index in range(0, len(sentences), self.max_sentences_per_chunk)
        ]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return self._split(text.strip(), self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        current_text = current_text.strip()
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # No separator left: a fixed-size split is the safe fallback.
        if not remaining_separators:
            return [
                current_text[index : index + self.chunk_size].strip()
                for index in range(0, len(current_text), self.chunk_size)
                if current_text[index : index + self.chunk_size].strip()
            ]

        separator = remaining_separators[0]
        later_separators = remaining_separators[1:]
        if separator == "":
            return [
                current_text[index : index + self.chunk_size].strip()
                for index in range(0, len(current_text), self.chunk_size)
                if current_text[index : index + self.chunk_size].strip()
            ]
        if separator not in current_text:
            return self._split(current_text, later_separators)

        raw_parts = current_text.split(separator)
        parts = [
            (part + separator if index < len(raw_parts) - 1 else part).strip()
            for index, part in enumerate(raw_parts)
            if part.strip()
        ]

        chunks: list[str] = []
        pending = ""
        for part in parts:
            if len(part) > self.chunk_size:
                if pending:
                    chunks.append(pending)
                    pending = ""
                chunks.extend(self._split(part, later_separators))
            elif not pending:
                pending = part
            elif len(pending) + 1 + len(part) <= self.chunk_size:
                pending = f"{pending} {part}"
            else:
                chunks.append(pending)
                pending = part

        if pending:
            chunks.append(pending)
        return chunks


class HeadingChunker:
    """Chunk Markdown by headings, preserving heading context in long sections."""

    def __init__(self, chunk_size: int = 800) -> None:
        self.chunk_size = chunk_size
        self._fallback = RecursiveChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sections = [section.strip() for section in re.split(r"(?=^#{1,6}\s+)", text.strip(), flags=re.MULTILINE) if section.strip()]
        document_heading = ""
        chunks: list[str] = []

        for section in sections:
            heading_match = re.match(r"^(#{1,6}\s+[^\n]+)", section)
            heading = heading_match.group(1).strip() if heading_match else ""
            if heading.startswith("# "):
                document_heading = heading

            if len(section) <= self.chunk_size:
                chunks.append(section)
                continue

            context_parts = [part for part in (document_heading, heading) if part]
            context = "\n\n".join(dict.fromkeys(context_parts))
            available_size = max(1, self.chunk_size - len(context) - 2)
            subchunks = RecursiveChunker(chunk_size=available_size).chunk(section)
            chunks.extend(
                f"{context}\n\n{subchunk}".strip() if context else subchunk
                for subchunk in subchunks
            )

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    magnitude_a = math.sqrt(_dot(vec_a, vec_a))
    magnitude_b = math.sqrt(_dot(vec_b, vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return _dot(vec_a, vec_b) / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        overlap = min(50, max(0, chunk_size - 1))
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=overlap),
            "by_sentences": SentenceChunker(),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }
        comparison = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            comparison[name] = {
                "count": len(chunks),
                "avg_length": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0.0,
                "chunks": chunks,
            }
        return comparison
