from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from convert_codex_calls_to_mooncake_trace import (
    convert_codex_calls_to_mooncake,
)


class ConvertCodexCallsToMooncakeTraceTest(unittest.TestCase):
    def test_converts_lengths_timing_and_synthetic_cache_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "model_calls.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "arrival_time_s": 10.0,
                            "input_tokens": 1200,
                            "cached_input_tokens": 0,
                            "output_tokens": 20,
                            "model": "deepseek-v4-flash",
                            "model_provider": "deepseek",
                            "rollout_id": "rollout-a",
                        },
                        {
                            "arrival_time_s": 10.125,
                            "input_tokens": 1300,
                            "cached_input_tokens": 1024,
                            "output_tokens": 30,
                            "model": "deepseek-v4-flash",
                            "model_provider": "deepseek",
                            "rollout_id": "rollout-a",
                        },
                        {
                            "arrival_time_s": 11.0,
                            "input_tokens": 700,
                            "cached_input_tokens": 512,
                            "output_tokens": 10,
                            "model": "other-model",
                            "model_provider": "other-provider",
                            "rollout_id": "rollout-b",
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "trace"
            manifest = convert_codex_calls_to_mooncake(
                source, output, compress=False
            )
            rows = _load_jsonl(output / "mooncake_trace.jsonl")

        self.assertEqual([row["timestamp"] for row in rows], [0, 125, 1000])
        self.assertEqual(rows[0]["input_length"], 1200)
        self.assertEqual(rows[1]["output_length"], 30)
        self.assertEqual([len(row["hash_ids"]) for row in rows], [3, 3, 2])
        self.assertEqual(rows[0]["hash_ids"], [0, 1, 2])
        self.assertEqual(rows[1]["hash_ids"][:2], [0, 1])
        self.assertNotEqual(rows[1]["hash_ids"][0], rows[2]["hash_ids"][0])
        self.assertEqual(manifest["counts"]["requests"], 3)
        self.assertEqual(manifest["counts"]["represented_cached_blocks"], 3)
        self.assertEqual(manifest["counts"]["replayable_cached_blocks"], 2)
        self.assertEqual(manifest["counts"]["span_ms"], 1000)
        self.assertNotIn("prompt", rows[0])

    def test_refuses_nonmonotonic_arrivals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "model_calls.jsonl"
            source.write_text(
                json.dumps(_call(2.0)) + "\n" + json.dumps(_call(1.0)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not monotonic"):
                convert_codex_calls_to_mooncake(
                    source, root / "trace", compress=False
                )

    def test_refuses_existing_output_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "model_calls.jsonl"
            source.write_text(json.dumps(_call(0.0)) + "\n", encoding="utf-8")
            output = root / "trace"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                convert_codex_calls_to_mooncake(source, output, compress=False)


def _call(arrival: float) -> dict:
    return {
        "arrival_time_s": arrival,
        "input_tokens": 100,
        "cached_input_tokens": 0,
        "output_tokens": 10,
        "model": "model",
        "model_provider": "provider",
        "rollout_id": "rollout",
    }


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    unittest.main()
