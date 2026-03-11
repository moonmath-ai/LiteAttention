#!/usr/bin/env python3
"""Fail if any GitHub workflow introduces an automatic push trigger."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"


def _iter_workflows() -> Iterable[Path]:
    for pattern in ("*.yml", "*.yaml"):
        yield from sorted(WORKFLOW_DIR.glob(pattern))


def _normalize_on_value(raw_on):
    if isinstance(raw_on, dict):
        return raw_on
    if isinstance(raw_on, list):
        return [str(v).strip() for v in raw_on]
    if raw_on is None:
        return None
    return str(raw_on).strip()


def _has_push_trigger_yaml(path: Path) -> tuple[bool, str]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        return _has_push_trigger_regex(path), f"regex-fallback ({exc.__class__.__name__})"

    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except Exception as exc:
        return _has_push_trigger_regex(path), f"regex-fallback (yaml-parse-error: {exc.__class__.__name__})"

    if not isinstance(data, dict):
        return False, "yaml"

    on_value = data.get("on")
    on_norm = _normalize_on_value(on_value)

    if isinstance(on_norm, dict):
        if "push" in on_norm:
            return True, "yaml"
        return False, "yaml"
    if isinstance(on_norm, list):
        if "push" in on_norm:
            return True, "yaml"
        return False, "yaml"
    if isinstance(on_norm, str):
        if on_norm == "push":
            return True, "yaml"
        return False, "yaml"

    return False, "yaml"


def _has_push_trigger_regex(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    # Map style:
    # on:
    #   push:
    if re.search(r"(?m)^\s*push\s*:\s*(?:#.*)?$", text):
        return True
    # Sequence style:
    # on: [push, pull_request]
    if re.search(r"(?mi)^\s*on\s*:\s*\[[^\]]*\bpush\b[^\]]*\]\s*(?:#.*)?$", text):
        return True
    # Scalar style:
    # on: push
    if re.search(r"(?mi)^\s*on\s*:\s*push\s*(?:#.*)?$", text):
        return True
    return False


def main() -> int:
    offenders: list[tuple[Path, str]] = []
    for workflow in _iter_workflows():
        has_push, mode = _has_push_trigger_yaml(workflow)
        if has_push:
            offenders.append((workflow, mode))

    if offenders:
        print("ERROR: push-triggered workflow(s) detected:")
        for wf, mode in offenders:
            print(f"  - {wf.relative_to(ROOT)} ({mode})")
        print("Refusing: pushes to branch-only workflows must remain non-triggering.")
        return 1

    print("OK: no workflow contains an on: push trigger.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
