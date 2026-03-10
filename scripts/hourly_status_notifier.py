#!/usr/bin/env python3
"""One-shot hourly plain-text notifier for LiteAttention benchmark status."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html as html_lib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STATUS_FILE = Path("/tmp/liteattention_status.txt")
DEFAULT_STATE_FILE = Path("/tmp/liteattention_notify_state.json")
DEFAULT_LOCK_FILE = Path("/tmp/liteattention_notify_once.lock")
PROTON_DIR = Path("/root/.codex/skills/proton-notify")
PROTON_SCRIPT = PROTON_DIR / "scripts" / "proton_notify.js"
PROTON_CREDS = PROTON_DIR / "creds.txt"
TEXT_WIDTH = 78


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(dt: datetime | None = None) -> str:
    return (dt or utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def hour_bucket(dt: datetime | None = None) -> str:
    return (dt or utc_now()).strftime("%Y-%m-%dT%H:00Z")


def short_hour_label(dt: datetime | None = None) -> str:
    return (dt or utc_now()).strftime("%Y-%m-%d %H:00Z")


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[0-9]+\.\s*", "", text)
    text = re.sub(r"^-\s*", "", text)
    return " ".join(text.split())


def fill_text(text: str, *, initial_indent: str = "", subsequent_indent: str = "") -> str:
    return textwrap.fill(
        clean_text(text),
        width=TEXT_WIDTH,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def render_bullets(items: list[str]) -> list[str]:
    lines: list[str] = []
    for item in items:
        lines.append(fill_text(item, initial_indent="- ", subsequent_indent="  "))
    return lines


def parse_status(text: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    kv: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^([A-Za-z0-9_]+)=(.*)$", stripped)
        if match:
            kv[match.group(1)] = match.group(2).strip()
        if current:
            if stripped.endswith(":") and not stripped.startswith("-"):
                current = stripped[:-1]
                sections.setdefault(current, [])
            else:
                sections.setdefault(current, []).append(line)
            continue
        if match:
            continue
        if stripped.endswith(":") and not stripped.startswith("-"):
            current = stripped[:-1]
            sections.setdefault(current, [])
    return kv, sections


def extract_active_lanes(section_lines: list[str]) -> list[tuple[str, str]]:
    lanes: list[tuple[str, str]] = []
    in_lanes = False
    for raw in section_lines:
        stripped = raw.strip()
        if stripped == "- active implementation/research lanes:":
            in_lanes = True
            continue
        if not in_lanes:
            continue
        if stripped.startswith("- ") and "=" not in stripped:
            break
        if "=" in stripped:
            key, value = stripped.split("=", 1)
            lanes.append((key.strip(), clean_text(value)))
    return lanes


def extract_experiment_bullets(section_lines: list[str]) -> list[str]:
    bullets: list[str] = []
    current_bullet = ""
    for raw in section_lines:
        stripped = raw.strip()
        if stripped == "- active implementation/research lanes:":
            break
        if stripped.startswith("- "):
            if current_bullet:
                bullets.append(current_bullet)
            current_bullet = clean_text(stripped)
        elif current_bullet and stripped:
            current_bullet = f"{current_bullet} {clean_text(stripped)}"
    if current_bullet:
        bullets.append(current_bullet)
    return bullets


def extract_future_ideas(section_lines: list[str]) -> list[str]:
    ideas: list[str] = []
    for raw in section_lines:
        stripped = raw.strip()
        if not stripped:
            continue
        ideas.append(clean_text(stripped))
    return ideas


def extract_percent(best_header: str, speedup_text: str) -> str:
    for source in (best_header, speedup_text):
        match = re.search(r"([+-]?\d+(?:\.\d+)?%)", source)
        if match:
            value = match.group(1)
            return value if value.startswith(("+", "-")) else f"+{value}"
    return "n/a"


def extract_eta(eta_text: str, best_header: str) -> str:
    eta_text = clean_text(eta_text)
    if eta_text:
        return eta_text
    if best_header:
        head = best_header.split("|", 1)[0].strip()
        return head.removeprefix("eta ").strip() or "unknown"
    return "unknown"


def short_eta(eta_text: str) -> str:
    return eta_text.split(" with", 1)[0].strip()


def infer_blocker(experiment_bullets: list[str], lanes: list[tuple[str, str]]) -> str:
    haystack = experiment_bullets + [value for _, value in lanes]
    for item in haystack:
        lowered = item.lower()
        if any(token in lowered for token in (" blocker", " blocked", " failed", " failure", " error", " unavailable", " stuck")):
            return item
    return "none"


def build_payload(status_text: str, status_exists: bool) -> dict[str, Any]:
    kv, sections = parse_status(status_text)
    running = sections.get("current_running_experiments", [])
    future = sections.get("future_plan", [])
    lanes = extract_active_lanes(running)
    experiment_bullets = extract_experiment_bullets(running)
    future_ideas = extract_future_ideas(future)

    eta_text = extract_eta(
        kv.get("best_projected_whole_matrix_eta") or kv.get("whole_matrix_eta") or "",
        kv.get("best_matrix_header", ""),
    )
    best_header = kv.get("best_matrix_header", "unavailable")
    speedup = extract_percent(best_header, kv.get("whole_matrix_speedup_vs_unoptimized", ""))
    focus_tag = lanes[0][0].replace("_", "-") if lanes else "status"
    primary_task = lanes[0][1] if lanes else "maintain benchmark coordination"
    secondary_tasks = [value for _, value in lanes[1:3]]
    ideas = future_ideas[:4] + [f"{key.replace('_', ' ')}: {value}" for key, value in lanes[:2]]
    if not ideas:
        ideas = [
            "keep workflow work moving",
            "refresh the status payload",
            "verify the sender path",
            "preserve benchmark coverage",
        ]
    return {
        "status_exists": status_exists,
        "best_header": best_header,
        "eta": eta_text,
        "eta_short": short_eta(eta_text),
        "speedup": speedup,
        "branch": kv.get("main_branch", "unknown"),
        "head": kv.get("head", "unknown"),
        "focus_tag": focus_tag,
        "primary_task": primary_task,
        "secondary_tasks": secondary_tasks,
        "experiment_bullets": experiment_bullets[:5],
        "ideas": ideas[:6],
        "top_blocker": infer_blocker(experiment_bullets, lanes),
    }


def build_delta_lines(current: dict[str, Any], previous: dict[str, Any] | None) -> list[str]:
    if not previous:
        return ["First hourly status from the repo-owned notifier."]

    lines: list[str] = []
    if current["primary_task"] != previous.get("primary_task"):
        lines.append(f"Primary task: {current['primary_task']}")
    if current["best_header"] != previous.get("best_header"):
        lines.append(f"ETA / speedup: {current['best_header']}")
    branch_head = f"{current['branch']} @ {current['head']}"
    prev_branch_head = f"{previous.get('branch', 'unknown')} @ {previous.get('head', 'unknown')}"
    if branch_head != prev_branch_head:
        lines.append(f"Branch/head: {branch_head}")
    if current["top_blocker"] != previous.get("top_blocker"):
        lines.append(f"Top blocker: {current['top_blocker']}")

    prev_bullets = set(previous.get("experiment_bullets", []))
    for bullet in current["experiment_bullets"]:
        if bullet not in prev_bullets:
            lines.append(bullet)
        if len(lines) >= 4:
            break

    return lines[:4] or ["No material change in the last hour."]


def tldr_line(payload: dict[str, Any], dt: datetime) -> str:
    templates = [
        "TL;DR: Main work is {primary}. ETA stays {eta}.",
        "TL;DR: Current focus is {focus}. Main risk is {blocker}.",
        "TL;DR: This hour is on {primary}. Best estimate remains {eta}.",
        "TL;DR: Work remains centered on {focus}. The current constraint is {blocker}.",
    ]
    template = templates[dt.hour % len(templates)]
    return template.format(
        primary=payload["primary_task"],
        eta=payload["eta_short"],
        focus=payload["focus_tag"],
        blocker=payload["top_blocker"],
    )


def build_subject(payload: dict[str, Any], dt: datetime) -> str:
    return (
        f"LiteAttention | ETA {payload['eta_short']} | {payload['speedup']} | "
        f"{payload['focus_tag']} | {short_hour_label(dt)}"
    )


def build_body(payload: dict[str, Any], delta_lines: list[str], dt: datetime) -> str:
    lines = [
        "LiteAttention hourly status",
        f"ETA / speedup: {payload['eta_short']} | {payload['speedup']}",
        "",
        fill_text(tldr_line(payload, dt)),
        "",
        "Doing now",
    ]
    lines.extend(render_bullets([payload["primary_task"], *payload["secondary_tasks"][:2]]))

    lines.extend(["", "What changed"])
    lines.extend(render_bullets(delta_lines[:4]))

    lines.extend(
        [
            "",
            "ETA / risk",
        ]
    )
    lines.extend(
        render_bullets(
            [
                f"Current-task ETA: {payload['eta']}",
                f"ETA/% header: {payload['best_header']}",
                f"Top blocker: {payload['top_blocker']}",
            ]
        )
    )

    lines.extend(
        [
            "",
            "Idea list",
        ]
    )
    lines.extend(
        render_bullets(["; ".join(payload["ideas"][:6])])
    )

    branch_head = f"{payload['branch']} @ {payload['head']}"
    if delta_lines != ["No material change in the last hour."] or not payload["status_exists"]:
        lines.extend(["", "Meta"])
        lines.extend(render_bullets([f"Branch/head: {branch_head}"]))
    return "\n".join(lines).strip() + "\n"


def build_html_preview(payload: dict[str, Any], delta_lines: list[str], dt: datetime) -> str:
    def li(items: list[str]) -> str:
        return "".join(f"<li>{html_lib.escape(clean_text(item))}</li>" for item in items)

    return f"""\
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>LiteAttention hourly status</title>
  </head>
  <body style="margin:0;padding:0;background:#f5f7fb;color:#122033;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
    <div style="max-width:640px;margin:0 auto;padding:16px;">
      <div style="background:#ffffff;border-radius:18px;padding:18px;box-shadow:0 6px 24px rgba(18,32,51,0.08);">
        <div style="font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#5b6b82;margin-bottom:8px;">LiteAttention hourly status</div>
        <div style="font-size:15px;line-height:1.45;margin-bottom:12px;">ETA / speedup: {html_lib.escape(payload['eta_short'])} | {html_lib.escape(payload['speedup'])}</div>
        <div style="font-size:18px;line-height:1.35;font-weight:700;margin-bottom:14px;">{html_lib.escape(tldr_line(payload, dt))}</div>

        <div style="margin-bottom:14px;">
          <div style="font-size:12px;font-weight:700;text-transform:uppercase;color:#5b6b82;margin-bottom:6px;">Doing now</div>
          <ul style="margin:0;padding-left:18px;line-height:1.5;">
            {li([payload['primary_task'], *payload['secondary_tasks'][:2]])}
          </ul>
        </div>

        <div style="margin-bottom:14px;">
          <div style="font-size:12px;font-weight:700;text-transform:uppercase;color:#5b6b82;margin-bottom:6px;">What changed</div>
          <ul style="margin:0;padding-left:18px;line-height:1.5;">
            {li(delta_lines[:4])}
          </ul>
        </div>

        <div style="margin-bottom:14px;">
          <div style="font-size:12px;font-weight:700;text-transform:uppercase;color:#5b6b82;margin-bottom:6px;">ETA / risk</div>
          <ul style="margin:0;padding-left:18px;line-height:1.5;">
            {li([f"Current-task ETA: {payload['eta']}", f"ETA/% header: {payload['best_header']}", f"Top blocker: {payload['top_blocker']}"])}
          </ul>
        </div>

        <div style="margin-bottom:14px;">
          <div style="font-size:12px;font-weight:700;text-transform:uppercase;color:#5b6b82;margin-bottom:6px;">Idea list</div>
          <div style="font-size:13px;line-height:1.55;color:#22344f;">{html_lib.escape('; '.join(payload['ideas'][:6]))}</div>
        </div>

        <div style="font-size:12px;color:#5b6b82;">Branch/head: {html_lib.escape(payload['branch'])} @ {html_lib.escape(payload['head'])}</div>
      </div>
    </div>
  </body>
</html>
""".strip() + "\n"


def fallback_body(payload: dict[str, Any]) -> str:
    return (
        "LiteAttention hourly status\n\n"
        "TL;DR: notifier fallback mode is active. Current work is continuing, "
        "but the structured summary did not validate.\n\n"
        "Doing now\n"
        f"{fill_text(payload['primary_task'], initial_indent='- ', subsequent_indent='  ')}\n\n"
        "What changed\n"
        f"{fill_text('Fallback path used because the rendered body failed validation.', initial_indent='- ', subsequent_indent='  ')}\n\n"
        "ETA / risk\n"
        f"{fill_text(f'Current-task ETA: {payload['eta']}', initial_indent='- ', subsequent_indent='  ')}\n"
        f"{fill_text(f'ETA/% header: {payload['best_header']}', initial_indent='- ', subsequent_indent='  ')}\n"
        f"{fill_text('Top blocker: notifier render failure', initial_indent='- ', subsequent_indent='  ')}\n\n"
        "Idea list\n"
        f"{fill_text('restore payload quality; verify sender health; keep hourly delivery alive', initial_indent='- ', subsequent_indent='  ')}\n"
    )


def validate_subject(subject: str) -> list[str]:
    errors: list[str] = []
    stripped = subject.strip()
    if not stripped:
        errors.append("subject is empty")
    if len(stripped) > 200:
        errors.append("subject is too long")
    if "ETA " not in subject:
        errors.append("subject missing ETA")
    if "|" not in subject:
        errors.append("subject missing separators")
    return errors


def validate_body(body: str) -> list[str]:
    errors: list[str] = []
    stripped = body.strip()
    if not stripped:
        errors.append("body is empty")
    if len(stripped) < 180:
        errors.append("body is too short")
    for marker in ("TL;DR:", "Doing now", "What changed", "ETA / risk", "Idea list"):
        if marker not in body:
            errors.append(f"missing section: {marker}")
    return errors


def canonical_hash(data: dict[str, Any]) -> str:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def load_proton_env() -> dict[str, str]:
    env = os.environ.copy()
    if env.get("PROTON_EMAIL") and env.get("PROTON_PASSWORD"):
        return env
    if not PROTON_CREDS.exists():
        return env
    email = ""
    password = ""
    for raw in PROTON_CREDS.read_text().splitlines():
        if raw.startswith("email:"):
            email = raw.split(":", 1)[1].strip()
        if raw.startswith("password:"):
            password = raw.split(":", 1)[1].strip()
    if email and password:
        env["PROTON_EMAIL"] = email
        env["PROTON_PASSWORD"] = password
    return env


def send_email(subject: str, body: str, timeout_seconds: int) -> tuple[bool, str]:
    env = load_proton_env()
    cmd = [
        "node",
        str(PROTON_SCRIPT),
        "send",
        "--subject",
        subject,
        "--body",
        body,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROTON_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"send timeout after {timeout_seconds}s: {exc}"
    except OSError as exc:
        return False, f"send launch failed: {exc}"

    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0 and "Send result: success" in output, output


def acquire_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"lock busy: {path}", file=sys.stderr)
        sys.exit(0)
    return handle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_FILE)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK_FILE)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--html-preview-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    lock_handle = acquire_lock(args.lock_file)
    _ = lock_handle

    now = utc_now()
    now_iso = utc_iso(now)
    now_hour = hour_bucket(now)

    state = load_state(args.state_file)
    if (
        not args.force
        and not args.dry_run
        and state.get("last_sent_hour_utc") == now_hour
        and state.get("last_result") == "success"
    ):
        print(f"already sent for {now_hour}")
        return 0

    status_exists = args.status_file.exists()
    status_text = args.status_file.read_text() if status_exists else ""
    payload = build_payload(status_text, status_exists)
    previous_payload = state.get("snapshot")
    delta_lines = build_delta_lines(payload, previous_payload)
    subject = build_subject(payload, now)
    body = build_body(payload, delta_lines, now)
    subject_errors = validate_subject(subject)
    if subject_errors:
        print(f"subject validation failed: {subject_errors}", file=sys.stderr)
        return 1
    validation_errors = validate_body(body)
    if validation_errors:
        body = fallback_body(payload)
        validation_errors = validate_body(body)
    if validation_errors:
        print(f"body validation failed: {validation_errors}", file=sys.stderr)
        return 1

    html_preview = build_html_preview(payload, delta_lines, now)
    if args.html_preview_file:
        args.html_preview_file.write_text(html_preview)

    snapshot = {
        "focus_tag": payload["focus_tag"],
        "primary_task": payload["primary_task"],
        "secondary_tasks": payload["secondary_tasks"],
        "best_header": payload["best_header"],
        "eta": payload["eta"],
        "speedup": payload["speedup"],
        "top_blocker": payload["top_blocker"],
        "experiment_bullets": payload["experiment_bullets"],
        "ideas": payload["ideas"],
        "branch": payload["branch"],
        "head": payload["head"],
        "delta_lines": delta_lines,
    }
    body_hash = canonical_hash(snapshot)

    if args.dry_run:
        print(subject)
        print("---")
        print(body, end="")
        return 0

    backoffs = [15, 45, 120]
    send_output = ""
    success = False
    for idx, delay in enumerate(backoffs, start=1):
        success, send_output = send_email(subject, body, args.timeout_seconds)
        if success:
            break
        if idx < len(backoffs):
            time.sleep(delay)

    new_state = {
        "last_attempted_at_utc": now_iso,
        "last_sent_hour_utc": now_hour if success else state.get("last_sent_hour_utc", ""),
        "last_sent_at_utc": now_iso if success else state.get("last_sent_at_utc", ""),
        "last_subject": subject,
        "last_body_hash": body_hash if success else state.get("last_body_hash", ""),
        "last_result": "success" if success else "failed",
        "last_error": "" if success else send_output[-4000:],
        "snapshot": snapshot if success else state.get("snapshot", {}),
    }
    save_state(args.state_file, new_state)

    if not success:
        print(send_output, file=sys.stderr)
        return 1

    print(send_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
