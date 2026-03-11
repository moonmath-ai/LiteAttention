#!/usr/bin/env python3
"""Summarize LiteAttention build-profile artifacts into a short text report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_summary_path(path_arg: str) -> Path:
    path = Path(path_arg)
    candidates: list[Path] = []
    if path.is_file():
        if path.name == "summary.json":
            return path
        raise FileNotFoundError(f"{path} is not a summary.json file")

    candidates.extend(
        [
            path / "summary.json",
            path / "build-profile" / "summary.json",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(path.glob("**/build-profile/summary.json"))
    if matches:
        return matches[0]
    matches = sorted(path.glob("**/summary.json"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"could not find summary.json under {path}")


def fmt_s(value: Any) -> str:
    try:
        return f"{float(value):.3f}s"
    except Exception:
        return "n/a"


def fmt_pct(value: float, total: float) -> str:
    if total <= 0:
        return "n/a"
    return f"{(100.0 * value / total):.2f}%"


def top_pairs(rows: list[Any], limit: int) -> list[str]:
    out: list[str] = []
    for row in rows[:limit]:
        if isinstance(row, dict):
            label = row.get("file") or row.get("output") or row.get("source") or "unknown"
            value = row.get("wall_s") or row.get("duration_s") or row.get("total_s")
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            label, value = row[0], row[1]
        else:
            continue
        out.append(f"{fmt_s(value)}  {label}")
    return out


def emit_report(summary: dict[str, Any], summary_path: Path) -> str:
    counts = summary.get("counts", {})
    runtime = summary.get("runtime_info", {})
    stage_totals_rows = summary.get("nvcc_stage_totals_s", []) or []
    stage_totals: dict[str, float] = {}
    for row in stage_totals_rows:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            try:
                stage_totals[str(row[0])] = float(row[1])
            except Exception:
                continue
    nvcc_total_s = sum(stage_totals.values())
    lines: list[str] = []

    lines.append(f"build-profile summary: {summary_path}")
    lines.append(
        "runtime: "
        f"nvcc={runtime.get('nvcc_path') or 'missing'}  "
        f"time={runtime.get('time_bin_kind') or 'unknown'}  "
        f"ccache_wrappers={runtime.get('use_ccache_wrappers', 'unknown')}"
    )
    lines.append(
        "coverage: "
        f"nvcc_time_files={counts.get('nvcc_time_files', 0)}  "
        f"normalized={counts.get('nvcc_time_files_normalized', 0)}  "
        f"ambiguous={counts.get('nvcc_time_files_ambiguous', 0)}  "
        f"rows={counts.get('nvcc_time_rows_written', 0)}"
    )
    lines.append(
        "totals: "
        f"setup_py={fmt_s((summary.get('setup_py_wall_s') or [{}])[0].get('wall_s')) if summary.get('setup_py_wall_s') else 'n/a'}  "
        f"ptxas={fmt_s(summary.get('ptxas_total_from_nvcc_time_s'))}  "
        f"strict_units={summary.get('nvcc_time_unit_normalization_strict', False)}"
    )
    phase_keys = ["cicc", "cudafe++", "gcc (compiling)", "ptxas"]
    present_phase_keys = [k for k in phase_keys if k in stage_totals]
    if present_phase_keys:
        parts = [
            f"{k}={fmt_s(stage_totals[k])} ({fmt_pct(stage_totals[k], nvcc_total_s)})"
            for k in present_phase_keys
        ]
        lines.append("nvcc phases: " + "  ".join(parts))

    top_setup = top_pairs(summary.get("setup_py_wall_s", []), 1)
    if top_setup:
        lines.append("top setup.py wall:")
        lines.extend(f"- {row}" for row in top_setup)

    top_tu = top_pairs(summary.get("top_tu_wall_s", []), 3)
    if top_tu:
        lines.append("top TU wall:")
        lines.extend(f"- {row}" for row in top_tu)

    top_stages = top_pairs(summary.get("nvcc_stage_totals_s", []), 5)
    if top_stages:
        lines.append("top NVCC stages:")
        lines.extend(f"- {row}" for row in top_stages)

    top_edges = top_pairs(summary.get("top_ninja_outputs_s", []), 3)
    if top_edges:
        lines.append("top ninja edges:")
        lines.extend(f"- {row}" for row in top_edges)

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="build-profile dir, summary.json, or run dir containing it")
    args = parser.parse_args()

    try:
        summary_path = find_summary_path(args.path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    summary = load_json(summary_path)
    print(emit_report(summary, summary_path), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
