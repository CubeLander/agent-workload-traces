from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prepare_codex_subagent_trace import (
    discover_selected_rollouts,
    export_codex_subagent_dataset,
    redact_text,
)


def _record(timestamp: str, record_type: str, payload: dict) -> str:
    return json.dumps(
        {"timestamp": timestamp, "type": record_type, "payload": payload}
    )


def _write_rollout(
    path: Path,
    *,
    role: str = "deepseek_worker",
    model: str = "combo/deepseek-v4-flash",
) -> None:
    rows = [
        _record(
            "2026-08-01T00:00:00Z",
            "session_meta",
            {
                "id": "rollout-1",
                "session_id": "parent-1",
                "parent_thread_id": "parent-1",
                "agent_path": "/root/example",
                "agent_role": role,
                "agent_nickname": "ChangingNickname",
                "model_provider": "deepseek",
                "thread_source": "subagent",
                "cwd": "/workspace/example",
            },
        ),
        _record(
            "2026-08-01T00:00:00.1Z",
            "event_msg",
            {"type": "task_started", "turn_id": "turn-1"},
        ),
        # Older compacted rollouts can embed parent metadata before the first
        # turn context. It must not replace the child rollout identity.
        _record(
            "2026-08-01T00:00:00.15Z",
            "session_meta",
            {
                "id": "parent-1",
                "agent_role": None,
                "thread_source": "user",
            },
        ),
        _record(
            "2026-08-01T00:00:00.2Z",
            "turn_context",
            {"model": model, "effort": "max"},
        ),
        _record(
            "2026-08-01T00:00:00.3Z",
            "inter_agent_communication_metadata",
            {"trigger_turn": True},
        ),
        _record(
            "2026-08-01T00:00:00.4Z",
            "response_item",
            {
                "type": "agent_message",
                "author": "/root",
                "recipient": "/root/example",
                "content": [
                    {
                        "type": "input_text",
                        "text": "inspect this password=hunter2",
                    }
                ],
            },
        ),
        _record(
            "2026-08-01T00:00:01Z",
            "event_msg",
            {
                "type": "token_count",
                "info": {
                    "model_context_window": 1000,
                    "last_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 80,
                        "cache_write_input_tokens": 0,
                        "output_tokens": 10,
                        "reasoning_output_tokens": 4,
                        "total_tokens": 110,
                    },
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 80,
                        "cache_write_input_tokens": 0,
                        "output_tokens": 10,
                        "reasoning_output_tokens": 4,
                        "total_tokens": 110,
                    },
                },
            },
        ),
        # Codex can emit the same cumulative usage observation twice.
        _record(
            "2026-08-01T00:00:01.1Z",
            "event_msg",
            {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 80,
                        "cache_write_input_tokens": 0,
                        "output_tokens": 10,
                        "reasoning_output_tokens": 4,
                        "total_tokens": 110,
                    },
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 80,
                        "cache_write_input_tokens": 0,
                        "output_tokens": 10,
                        "reasoning_output_tokens": 4,
                        "total_tokens": 110,
                    },
                },
            },
        ),
        _record(
            "2026-08-01T00:00:01.2Z",
            "inter_agent_communication_metadata",
            {"trigger_turn": False},
        ),
        _record(
            "2026-08-01T00:00:01.3Z",
            "response_item",
            {
                "type": "agent_message",
                "author": "/root",
                "recipient": "/root/example",
                "content": [{"type": "input_text", "text": "one more check"}],
            },
        ),
        _record(
            "2026-08-01T00:00:02Z",
            "event_msg",
            {
                "type": "task_complete",
                "turn_id": "turn-1",
                "duration_ms": 1900,
                "time_to_first_token_ms": 20,
                "last_agent_message": "done",
            },
        ),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class PrepareCodexSubagentTraceTest(unittest.TestCase):
    def test_vera_selection_uses_role_and_model_not_nickname(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_rollout(root / "vera.jsonl")
            _write_rollout(
                root / "luna.jsonl", role="luna_worker", model="gpt-5.6-luna"
            )
            selected = discover_selected_rollouts((root,), selection="vera")
        self.assertEqual([item.path.name for item in selected], ["vera.jsonl"])

    def test_gpt_56_selection_accepts_family_and_provider_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_rollout(root / "sol.jsonl", role="worker", model="gpt-5.6-sol")
            _write_rollout(
                root / "luna.jsonl", role="luna_worker", model="openai/gpt-5.6-luna"
            )
            _write_rollout(root / "base.jsonl", role="worker", model="gpt-5.6")
            _write_rollout(root / "older.jsonl", role="worker", model="gpt-5.5")
            _write_rollout(root / "future.jsonl", role="worker", model="gpt-5.60-sol")
            selected = discover_selected_rollouts((root,), selection="gpt-5.6")
        self.assertEqual(
            [item.path.name for item in selected],
            ["base.jsonl", "luna.jsonl", "sol.jsonl"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            _write_rollout(source / "sol.jsonl", role="worker", model="gpt-5.6-sol")
            output = root / "dataset"
            export_codex_subagent_dataset(
                (source,), output, selection="gpt-5.6"
            )
            manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["selection"]["name"], "gpt-5.6")
        self.assertEqual(
            manifest["selection"]["predicate"]["model_basename_regex"],
            r"^gpt-5\.6(?:$|-)",
        )

    def test_exports_tasks_steering_calls_raw_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            _write_rollout(source / "vera.jsonl")
            output = root / "dataset"
            counts = export_codex_subagent_dataset(
                (source,), output, include_raw=True
            )

            tasks = _load_jsonl(output / "tasks.jsonl")
            steering = _load_jsonl(output / "steering.jsonl")
            calls = _load_jsonl(output / "model_calls.jsonl")
            manifest = json.loads((output / "manifest.json").read_text())

            self.assertEqual(counts["sessions"], 1)
            self.assertEqual(counts["tasks"], 1)
            self.assertEqual(counts["steering_messages"], 1)
            self.assertEqual(counts["model_calls"], 1)
            self.assertEqual(tasks[0]["input"], "inspect this password=<REDACTED>")
            self.assertEqual(tasks[0]["terminal_output"], "done")
            self.assertEqual(steering[0]["input"], "one more check")
            self.assertEqual(calls[0]["uncached_input_tokens"], 20)
            self.assertEqual(calls[0]["arrival_time_s"], 0.0)
            self.assertEqual(manifest["breakdown"]["models"], {"combo/deepseek-v4-flash": 1})
            self.assertEqual(
                manifest["privacy"]["curated_text_redaction_counts"]
                ["credential_assignment"],
                1,
            )
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            for path in output.rglob("*"):
                if path.is_file():
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            raw = list((output / "raw_rollouts").glob("*.jsonl"))
            self.assertEqual(len(raw), 1)
            self.assertIn("hunter2", raw[0].read_text())
            self.assertIn("raw_rollouts/", (output / "SHA256SUMS").read_text())

    def test_refuses_existing_output_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            _write_rollout(source / "vera.jsonl")
            output = root / "dataset"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                export_codex_subagent_dataset((source,), output)

    def test_redaction_handles_headers_and_credential_urls(self) -> None:
        text, counts = redact_text(
            "Authorization: Bearer abc123\nhttps://alice:secret@example.test/path"
        )
        self.assertNotIn("abc123", text)
        self.assertNotIn("secret", text)
        self.assertEqual(counts["authorization_header"], 1)
        self.assertEqual(counts["credential_url"], 1)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    unittest.main()
