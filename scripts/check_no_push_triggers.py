#!/usr/bin/env python3
"""Fail if any GitHub workflow introduces an automatic push trigger."""

from __future__ import annotations

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


def _has_push_trigger_yaml(path: Path) -> bool:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"{path.relative_to(ROOT)}: cannot import PyYAML ({exc.__class__.__name__})"
        ) from exc

    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except Exception as exc:
        raise RuntimeError(
            f"{path.relative_to(ROOT)}: YAML parse failed ({exc.__class__.__name__})"
        ) from exc

    if not isinstance(data, dict):
        return False

    on_value = data.get("on")
    on_norm = _normalize_on_value(on_value)

    if isinstance(on_norm, dict):
        if "push" in on_norm:
            return True
        return False
    if isinstance(on_norm, list):
        if "push" in on_norm:
            return True
        return False
    if isinstance(on_norm, str):
        if on_norm == "push":
            return True
        return False

    return False


def main() -> int:
    offenders: list[Path] = []
    errors: list[str] = []
    for workflow in _iter_workflows():
        try:
            has_push = _has_push_trigger_yaml(workflow)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        if has_push:
            offenders.append(workflow)

    if errors:
        print("ERROR: unable to validate workflow triggers safely:")
        for msg in errors:
            print(f"  - {msg}")
        print("Refusing: fix parser/import errors before trusting trigger checks.")
        return 2

    if offenders:
        print("ERROR: push-triggered workflow(s) detected:")
        for wf in offenders:
            print(f"  - {wf.relative_to(ROOT)}")
        print("Refusing: pushes to branch-only workflows must remain non-triggering.")
        return 1

    print("OK: no workflow contains an on: push trigger.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
