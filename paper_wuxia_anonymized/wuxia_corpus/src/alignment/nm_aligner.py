"""Self-contained monotonic N-M segment alignment."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Encoder(Protocol):
    """Minimal interface required from a sentence embedding model."""

    def encode(self, texts: Sequence[str], *, normalize_embeddings: bool) -> np.ndarray:
        ...


def _grouped_texts(segments: Sequence[str], maximum: int) -> dict[int, list[str]]:
    """Return every contiguous group of size 1 through ``maximum``."""
    return {
        size: [" ".join(segments[start : start + size]) for start in range(len(segments) - size + 1)]
        for size in range(1, min(maximum, len(segments)) + 1)
    }


def align_segments_nm(
    source_segments: Sequence[str],
    target_segments: Sequence[str],
    encoder: Encoder,
    *,
    max_source_group: int = 7,
    max_target_group: int = 7,
    skip_penalty: float = -0.5,
    group_size_penalty: float = 0.0,
) -> tuple[list[tuple[str, str]], list[float], dict[str, int]]:
    """Align two ordered segment sequences with unrestricted N-M transitions.

    A transition may consume any positive number of source and target segments
    up to the configured limits. Therefore, alignments such as 2-2, 2-3, and
    3-2 are considered in addition to the traditional 1-N and N-1 cases.
    Source-only and target-only transitions represent skipped segments.
    """
    if max_source_group < 1 or max_target_group < 1:
        raise ValueError("Alignment group limits must be positive integers.")

    source = list(source_segments)
    target = list(target_segments)
    source_groups = _grouped_texts(source, max_source_group)
    target_groups = _grouped_texts(target, max_target_group)

    source_vectors = {
        size: np.asarray(encoder.encode(texts, normalize_embeddings=True))
        for size, texts in source_groups.items()
    }
    target_vectors = {
        size: np.asarray(encoder.encode(texts, normalize_embeddings=True))
        for size, texts in target_groups.items()
    }
    source_count, target_count = len(source), len(target)
    scores = np.full((source_count + 1, target_count + 1), -np.inf)
    scores[0, 0] = 0.0
    backpointers: list[list[tuple[str, int, int, int, int, float] | None]] = [
        [None] * (target_count + 1) for _ in range(source_count + 1)
    ]

    for source_end in range(source_count + 1):
        for target_end in range(target_count + 1):
            for source_size, source_matrix in source_vectors.items():
                if source_end < source_size:
                    continue
                for target_size, target_matrix in target_vectors.items():
                    if target_end < target_size:
                        continue
                    source_start = source_end - source_size
                    target_start = target_end - target_size
                    similarity = float(
                        source_matrix[source_start] @ target_matrix[target_start]
                    )
                    transition = similarity - group_size_penalty * (
                        source_size + target_size - 2
                    )
                    candidate = scores[source_start, target_start] + transition
                    if candidate > scores[source_end, target_end]:
                        scores[source_end, target_end] = candidate
                        backpointers[source_end][target_end] = (
                            "align", source_start, target_start,
                            source_size, target_size, similarity,
                        )

            if source_end > 0:
                candidate = scores[source_end - 1, target_end] + skip_penalty
                if candidate > scores[source_end, target_end]:
                    scores[source_end, target_end] = candidate
                    backpointers[source_end][target_end] = (
                        "skip_source", source_end - 1, target_end, 1, 0, 0.0
                    )
            if target_end > 0:
                candidate = scores[source_end, target_end - 1] + skip_penalty
                if candidate > scores[source_end, target_end]:
                    scores[source_end, target_end] = candidate
                    backpointers[source_end][target_end] = (
                        "skip_target", source_end, target_end - 1, 0, 1, 0.0
                    )

    aligned: list[tuple[str, str]] = []
    aligned_scores: list[float] = []
    statistics: dict[str, int] = {
        "skip_ch": 0,
        "skip_en": 0,
        "total_segments_ch": source_count,
        "total_segments_en": target_count,
    }
    source_end, target_end = source_count, target_count
    while source_end > 0 or target_end > 0:
        step = backpointers[source_end][target_end]
        if step is None:
            raise RuntimeError("The dynamic-programming path could not be reconstructed.")
        action, source_start, target_start, source_size, target_size, similarity = step
        if action == "align":
            key = f"{source_size}-{target_size}"
            statistics[key] = statistics.get(key, 0) + 1
            aligned.append((
                " ".join(source[source_start:source_end]),
                " ".join(target[target_start:target_end]),
            ))
            aligned_scores.append(similarity)
        elif action == "skip_source":
            statistics["skip_ch"] += 1
        else:
            statistics["skip_en"] += 1
        source_end, target_end = source_start, target_start

    aligned.reverse()
    aligned_scores.reverse()
    return aligned, aligned_scores, statistics


def empty_alignment_statistics(max_source_group: int = 7, max_target_group: int = 7) -> dict[str, int]:
    """Create counters for every supported transition and bookkeeping field."""
    statistics = {
        f"{source_size}-{target_size}": 0
        for source_size in range(1, max_source_group + 1)
        for target_size in range(1, max_target_group + 1)
    }
    statistics.update({
        "skip_ch": 0,
        "skip_en": 0,
        "total_segments_ch": 0,
        "total_segments_en": 0,
    })
    return statistics


def print_alignment_metrics(statistics: dict[str, int], elapsed_seconds: float) -> None:
    """Print a compact summary for dynamically named N-M transitions."""
    source_total = statistics.get("total_segments_ch", 0)
    target_total = statistics.get("total_segments_en", 0)
    source_used = 0
    target_used = 0
    aligned_pairs = 0

    print("\n## Global alignment statistics")
    print("-" * 60)
    print(f"Processing time: {elapsed_seconds:.2f} seconds")
    print(f"Total Chinese segments: {source_total}")
    print(f"Total English segments: {target_total}")
    print("\n# Alignment actions")
    for key, count in sorted(statistics.items()):
        if not count or "-" not in key or key.startswith("skip"):
            continue
        source_size, target_size = map(int, key.split("-"))
        aligned_pairs += count
        source_used += source_size * count
        target_used += target_size * count
        print(f"{key} (Chinese: {source_size}, English: {target_size}): {count}")

    source_skips = statistics.get("skip_ch", 0)
    target_skips = statistics.get("skip_en", 0)
    print(f"Total aligned pairs: {aligned_pairs}")
    print(f"Skipped Chinese segments: {source_skips}")
    print(f"Skipped English segments: {target_skips}")
    source_unaccounted = source_total - source_used - source_skips
    target_unaccounted = target_total - target_used - target_skips
    if source_unaccounted or target_unaccounted:
        print(
            "Unaccounted segments: "
            f"Chinese={source_unaccounted}, English={target_unaccounted}"
        )
    print("-" * 60)
