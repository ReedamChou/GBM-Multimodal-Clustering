from __future__ import annotations

import re
from pathlib import Path


ITERATION_PATTERN = re.compile(r"^iteration-(\d+)$")


def list_iteration_dirs(results_root: Path) -> list[tuple[int, Path]]:
    iterations: list[tuple[int, Path]] = []
    if not results_root.exists():
        return iterations

    for child in results_root.iterdir():
        if not child.is_dir():
            continue
        match = ITERATION_PATTERN.match(child.name)
        if match is None:
            continue
        iterations.append((int(match.group(1)), child))
    return sorted(iterations, key=lambda item: item[0])


def latest_iteration_info(results_root: Path) -> tuple[int, Path] | None:
    iterations = list_iteration_dirs(results_root)
    if not iterations:
        return None
    return iterations[-1]


def latest_iteration_dir(results_root: Path) -> Path | None:
    info = latest_iteration_info(results_root)
    return None if info is None else info[1]


def next_iteration_info(results_root: Path) -> tuple[int, Path]:
    latest = latest_iteration_info(results_root)
    next_number = 1 if latest is None else latest[0] + 1
    return next_number, results_root / f"iteration-{next_number}"


def next_iteration_dir(results_root: Path) -> Path:
    return next_iteration_info(results_root)[1]
