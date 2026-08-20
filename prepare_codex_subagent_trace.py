#!/usr/bin/env python3
"""Extract private Codex subagent rollouts into an agent-workload dataset.

The source rollouts are append-only JSONL files written under ``~/.codex``.
This exporter deliberately keeps two representations separate:

* curated task, steering, and model-call tables for workload analysis; and
* optional verbatim rollout snapshots for future lossless reconstruction.

The curated model-call table is a traffic-shape contract.  A Codex rollout does
not contain a standalone, ready-to-send prompt for every model call, so this
script never pretends that a short task assignment is the full model context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOTS = (
    Path.home() / ".codex" / "sessions",
    Path.home() / ".codex" / "archived_sessions",
)
VERA_ROLE = "deepseek_worker"
VERA_MODEL_SUFFIX = "deepseek-v4-flash"
GPT_56_MODEL = re.compile(r"^gpt-5\.6(?:$|-)")
GPT_56_MODEL_PATTERN = GPT_56_MODEL.pattern

_SECRET_PATTERNS = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
            r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "authorization_header",
        re.compile(
            r"(?im)^(\s*authorization\s*:\s*)(?:bearer\s+|basic\s+)?[^\s]+\s*$"
        ),
    ),
    (
        "credential_assignment",
        re.compile(
            r"(?im)(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd)"
            r"\b\s*[:=]\s*)([^\s,;]+)"
        ),
    ),
    (
        "credential_url",
        re.compile(r"(?i)(https?://[^\s/:@]+:)([^\s/@]+)(@)"),
    ),
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    roots = tuple(path.expanduser().resolve() for path in args.input_root)
    result = export_codex_subagent_dataset(
        roots,
        args.output.expanduser().resolve(),
        selection=args.selection,
        include_raw=args.include_raw,
        max_sessions=args.max_sessions,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def export_codex_subagent_dataset(
    input_roots: Iterable[Path],
    output_dir: Path,
    *,
    selection: str = "vera",
    include_raw: bool = False,
    max_sessions: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a deterministic snapshot from the selected rollout files."""

    roots = tuple(Path(root).resolve() for root in input_roots)
    sources = discover_selected_rollouts(roots, selection=selection)
    if max_sessions is not None:
        sources = sources[:max_sessions]
    if not sources:
        raise ValueError(f"No matching Codex rollouts found for selection={selection!r}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, mode=0o700)

    sessions: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    steering: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    redactions: Counter[str] = Counter()
    earliest_epoch: float | None = None

    raw_dir = output_dir / "raw_rollouts"
    if include_raw:
        raw_dir.mkdir(mode=0o700)

    for source_index, source in enumerate(sources):
        rollout_id = str(source.metadata.get("id") or source.path.stem)
        if include_raw:
            raw_name = f"{source_index:04d}-{rollout_id}-{source.path.name}"
            snapshot_path = raw_dir / raw_name
        else:
            raw_name = ""
            snapshot_path = output_dir / ".rollout-snapshot.tmp"
        # Read the live store once. The copied prefix is a coherent input even
        # if Codex appends to the source while this snapshot is being built.
        shutil.copyfile(source.path, snapshot_path)
        snapshot_path.chmod(0o600)

        parsed = parse_rollout(
            snapshot_path,
            source.metadata,
            source.turn_context,
            source_path=source.path,
        )
        sessions.append(parsed["session"])
        tasks.extend(parsed["tasks"])
        steering.extend(parsed["steering"])
        model_calls.extend(parsed["model_calls"])

        for collection in (parsed["tasks"], parsed["steering"]):
            for row in collection:
                for key in ("input", "terminal_output"):
                    if isinstance(row.get(key), str):
                        row[key], counts = redact_text(row[key])
                        redactions.update(counts)
                row["sanitized"] = True

        for row in parsed["model_calls"]:
            epoch = row["_arrival_epoch"]
            earliest_epoch = epoch if earliest_epoch is None else min(earliest_epoch, epoch)

        if include_raw:
            parsed["session"]["raw_snapshot"] = f"raw_rollouts/{raw_name}"
        else:
            snapshot_path.unlink()

    if earliest_epoch is None:
        earliest_epoch = 0.0
    for row in model_calls:
        row["arrival_time_s"] = round(row.pop("_arrival_epoch") - earliest_epoch, 6)

    tasks.sort(key=_row_order)
    steering.sort(key=_row_order)
    model_calls.sort(key=_row_order)
    sessions.sort(key=lambda row: (row["started_at"], row["rollout_id"]))

    _write_jsonl(output_dir / "sessions.jsonl", sessions)
    _write_jsonl(output_dir / "tasks.jsonl", tasks)
    _write_jsonl(output_dir / "steering.jsonl", steering)
    _write_jsonl(output_dir / "model_calls.jsonl", model_calls)

    manifest = build_manifest(
        roots=roots,
        output_dir=output_dir,
        selection=selection,
        include_raw=include_raw,
        sessions=sessions,
        tasks=tasks,
        steering=steering,
        model_calls=model_calls,
        redactions=redactions,
    )
    _write_json(output_dir / "manifest.json", manifest)
    _write_checksums(output_dir)
    _make_private(output_dir)
    return manifest["counts"]


class SelectedRollout:
    def __init__(
        self,
        path: Path,
        metadata: dict[str, Any],
        turn_context: dict[str, Any],
    ) -> None:
        self.path = path
        self.metadata = metadata
        self.turn_context = turn_context


def discover_selected_rollouts(
    roots: Iterable[Path], *, selection: str
) -> list[SelectedRollout]:
    if selection not in {"vera", "gpt-5.6", "all-subagents"}:
        raise ValueError(f"Unsupported selection: {selection}")
    selected: list[SelectedRollout] = []
    paths = sorted(
        {
            path.resolve()
            for root in roots
            if root.exists()
            for path in root.rglob("*.jsonl")
            if path.is_file()
        }
    )
    for path in paths:
        metadata, turn_context = read_rollout_identity(path)
        if not metadata or metadata.get("thread_source") != "subagent":
            continue
        if selection == "vera" and not is_vera(metadata, turn_context):
            continue
        if selection == "gpt-5.6" and not is_gpt_56(turn_context):
            continue
        selected.append(SelectedRollout(path, metadata, turn_context))
    return selected


def read_rollout_identity(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    turn_context: dict[str, Any] = {}
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            if record.get("type") == "session_meta" and not metadata:
                metadata = payload
            elif record.get("type") == "turn_context" and not turn_context:
                turn_context = payload
            if metadata and turn_context:
                break
            if line_number >= 4096:
                break
    return metadata, turn_context


def is_vera(metadata: dict[str, Any], turn_context: dict[str, Any]) -> bool:
    model = str(turn_context.get("model") or "")
    return (
        metadata.get("agent_role") == VERA_ROLE
        and model.endswith(VERA_MODEL_SUFFIX)
    )


def is_gpt_56(turn_context: dict[str, Any]) -> bool:
    """Match the GPT-5.6 family, including provider-qualified model names."""

    model = str(turn_context.get("model") or "").rsplit("/", 1)[-1]
    return GPT_56_MODEL.match(model) is not None


def parse_rollout(
    path: Path,
    metadata: dict[str, Any],
    initial_turn_context: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    parse_errors = 0
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            if isinstance(record, dict):
                records.append(record)

    completions: dict[str, dict[str, Any]] = {}
    for record in records:
        payload = record.get("payload") or {}
        if record.get("type") == "event_msg" and payload.get("type") == "task_complete":
            turn_id = str(payload.get("turn_id") or "")
            completions[turn_id] = payload

    rollout_id = str(metadata.get("id") or path.stem)
    agent_path = str(metadata.get("agent_path") or "")
    model = str(initial_turn_context.get("model") or "")
    current_turn_id = ""
    current_turn_started_epoch: float | None = None
    pending_trigger: bool | None = None
    tasks: list[dict[str, Any]] = []
    steering: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    seen_usage_totals: set[tuple[int, ...]] = set()
    first_timestamp = ""
    last_timestamp = ""

    for event_index, record in enumerate(records):
        timestamp = str(record.get("timestamp") or "")
        if timestamp:
            first_timestamp = first_timestamp or timestamp
            last_timestamp = timestamp
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        record_type = record.get("type")
        payload_type = payload.get("type")

        if record_type == "event_msg" and payload_type == "task_started":
            current_turn_id = str(payload.get("turn_id") or "")
            current_turn_started_epoch = _timestamp_epoch(timestamp)
            continue
        if record_type == "inter_agent_communication_metadata":
            pending_trigger = bool(payload.get("trigger_turn"))
            continue
        if record_type == "response_item" and payload_type == "agent_message":
            content = _content_text(payload.get("content"))
            trigger_turn = pending_trigger
            pending_trigger = None
            if trigger_turn is None:
                trigger_turn = False
            completion = completions.get(current_turn_id, {})
            row = {
                "record_id": _stable_id(rollout_id, event_index, "task-input"),
                "rollout_id": rollout_id,
                "parent_thread_id": metadata.get("parent_thread_id"),
                "agent_path": agent_path,
                "agent_role": metadata.get("agent_role"),
                "agent_nickname": metadata.get("agent_nickname"),
                "model": model,
                "model_provider": metadata.get("model_provider"),
                "turn_id": current_turn_id,
                "trigger_turn": trigger_turn,
                "timestamp": timestamp,
                "event_index": event_index,
                "author": payload.get("author"),
                "recipient": payload.get("recipient"),
                "input": content,
                "terminal_output": completion.get("last_agent_message"),
                "terminal_status": "complete" if completion else "incomplete",
                "duration_ms": completion.get("duration_ms"),
                "time_to_first_token_ms": completion.get("time_to_first_token_ms"),
            }
            (tasks if trigger_turn else steering).append(row)
            continue
        if record_type == "event_msg" and payload_type == "token_count":
            info = payload.get("info")
            if not isinstance(info, dict):
                continue
            usage = info.get("last_token_usage")
            total_usage = info.get("total_token_usage")
            if not isinstance(usage, dict) or not isinstance(total_usage, dict):
                continue
            input_tokens = _nonnegative_int(usage.get("input_tokens"))
            output_tokens = _nonnegative_int(usage.get("output_tokens"))
            if input_tokens <= 0 or output_tokens <= 0:
                continue
            total_signature = tuple(
                _nonnegative_int(total_usage.get(key))
                for key in (
                    "input_tokens",
                    "cached_input_tokens",
                    "cache_write_input_tokens",
                    "output_tokens",
                    "reasoning_output_tokens",
                    "total_tokens",
                )
            )
            if total_signature in seen_usage_totals:
                continue
            seen_usage_totals.add(total_signature)
            epoch = _timestamp_epoch(timestamp)
            session_elapsed = (
                max(0.0, epoch - current_turn_started_epoch)
                if current_turn_started_epoch is not None
                else None
            )
            cached_tokens = _nonnegative_int(usage.get("cached_input_tokens"))
            calls.append(
                {
                    "request_id": _stable_id(rollout_id, event_index, "model-call"),
                    "rollout_id": rollout_id,
                    "agent_path": agent_path,
                    "agent_role": metadata.get("agent_role"),
                    "model": model,
                    "model_provider": metadata.get("model_provider"),
                    "turn_id": current_turn_id,
                    "timestamp": timestamp,
                    "session_elapsed_s": (
                        round(session_elapsed, 6) if session_elapsed is not None else None
                    ),
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached_tokens,
                    "uncached_input_tokens": max(0, input_tokens - cached_tokens),
                    "cache_write_input_tokens": _nonnegative_int(
                        usage.get("cache_write_input_tokens")
                    ),
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": _nonnegative_int(
                        usage.get("reasoning_output_tokens")
                    ),
                    "total_tokens": _nonnegative_int(usage.get("total_tokens")),
                    "model_context_window": info.get("model_context_window"),
                    "event_index": event_index,
                    "_arrival_epoch": epoch,
                }
            )

    session = {
        "rollout_id": rollout_id,
        "session_id": metadata.get("session_id"),
        "parent_thread_id": metadata.get("parent_thread_id"),
        "agent_path": agent_path,
        "agent_role": metadata.get("agent_role"),
        "agent_nickname": metadata.get("agent_nickname"),
        "model": model,
        "model_provider": metadata.get("model_provider"),
        "reasoning_effort": initial_turn_context.get("effort"),
        "originator": metadata.get("originator"),
        "cwd": metadata.get("cwd"),
        "started_at": first_timestamp,
        "ended_at": last_timestamp,
        "source_bytes": path.stat().st_size,
        "source_sha256": _sha256(path),
        "source_path": str(source_path or path),
        "parse_errors": parse_errors,
        "task_count": len(tasks),
        "steering_count": len(steering),
        "model_call_count": len(calls),
    }
    return {
        "session": session,
        "tasks": tasks,
        "steering": steering,
        "model_calls": calls,
    }


def redact_text(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    sanitized = text
    for label, pattern in _SECRET_PATTERNS:
        if label == "authorization_header":
            sanitized, count = pattern.subn(r"\1<REDACTED>", sanitized)
        elif label == "credential_assignment":
            sanitized, count = pattern.subn(r"\1<REDACTED>", sanitized)
        elif label == "credential_url":
            sanitized, count = pattern.subn(r"\1<REDACTED>\3", sanitized)
        else:
            sanitized, count = pattern.subn("<REDACTED_PRIVATE_KEY>", sanitized)
        counts[label] += count
    return sanitized, counts


def build_manifest(
    *,
    roots: tuple[Path, ...],
    output_dir: Path,
    selection: str,
    include_raw: bool,
    sessions: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    steering: list[dict[str, Any]],
    model_calls: list[dict[str, Any]],
    redactions: Counter[str],
) -> dict[str, Any]:
    model_counts = Counter(str(row.get("model")) for row in sessions)
    role_counts = Counter(str(row.get("agent_role")) for row in sessions)
    completed = sum(row["terminal_status"] == "complete" for row in tasks)
    selection_manifest: dict[str, Any] = {"name": selection}
    if selection == "vera":
        selection_manifest["predicate"] = {
            "agent_role": VERA_ROLE,
            "model_suffix": VERA_MODEL_SUFFIX,
        }
    elif selection == "gpt-5.6":
        selection_manifest["predicate"] = {
            "model_basename_regex": GPT_56_MODEL_PATTERN,
        }
    else:
        selection_manifest["predicate"] = {"thread_source": "subagent"}

    return {
        "schema_version": "codex-subagent-workload-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": selection_manifest,
        "source_roots": [str(root) for root in roots],
        "output_dir": str(output_dir),
        "include_raw_rollouts": include_raw,
        "privacy": {
            "directory_mode": "0700",
            "file_mode": "0600",
            "curated_text_redaction_counts": dict(sorted(redactions.items())),
            "raw_rollouts_are_verbatim_and_may_contain_secrets": include_raw,
        },
        "semantics": {
            "tasks": "trigger_turn=true inter-agent messages",
            "steering": "trigger_turn=false inter-agent messages",
            "model_calls": (
                "deduplicated valid token_count.last_token_usage observations; "
                "traffic shape, not reconstructed request bodies"
            ),
            "cached_input_tokens": (
                "provider-reported cache-hit portion of input_tokens"
            ),
        },
        "counts": {
            "sessions": len(sessions),
            "tasks": len(tasks),
            "completed_tasks": completed,
            "incomplete_tasks": len(tasks) - completed,
            "steering_messages": len(steering),
            "model_calls": len(model_calls),
            "source_bytes": sum(int(row["source_bytes"]) for row in sessions),
            "input_tokens": sum(int(row["input_tokens"]) for row in model_calls),
            "cached_input_tokens": sum(
                int(row["cached_input_tokens"]) for row in model_calls
            ),
            "uncached_input_tokens": sum(
                int(row["uncached_input_tokens"]) for row in model_calls
            ),
            "output_tokens": sum(int(row["output_tokens"]) for row in model_calls),
            "reasoning_output_tokens": sum(
                int(row["reasoning_output_tokens"]) for row in model_calls
            ),
        },
        "breakdown": {
            "models": dict(sorted(model_counts.items())),
            "roles": dict(sorted(role_counts.items())),
        },
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    path.chmod(0o600)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _write_checksums(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in paths
    ]
    target = output_dir / "SHA256SUMS"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    target.chmod(0o600)


def _make_private(output_dir: Path) -> None:
    for path in output_dir.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    output_dir.chmod(0o700)


def _content_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(parts)


def _timestamp_epoch(value: str) -> float:
    if not value:
        return 0.0
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(normalized).timestamp()


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _stable_id(rollout_id: str, event_index: int, kind: str) -> str:
    digest = hashlib.sha256(
        f"{rollout_id}\0{event_index}\0{kind}".encode("utf-8")
    ).hexdigest()[:24]
    return f"codex_{kind.replace('-', '_')}_{digest}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_order(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("timestamp") or ""),
        str(row.get("rollout_id") or ""),
        int(row.get("event_index") or 0),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Rollout root to scan. Repeat for multiple roots. Defaults to "
            "~/.codex/sessions and ~/.codex/archived_sessions."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "codex-vera-agent-workload",
    )
    parser.add_argument(
        "--selection",
        choices=("vera", "gpt-5.6", "all-subagents"),
        default="vera",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Copy selected rollouts verbatim into the private output snapshot.",
    )
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    args = parser.parse_args(argv)
    if args.input_root is None:
        args.input_root = list(DEFAULT_ROOTS)
    if args.max_sessions is not None and args.max_sessions <= 0:
        parser.error("--max-sessions must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
