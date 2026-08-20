#!/usr/bin/env python3
"""Convert content-free Codex model-call telemetry to a Mooncake-style trace.

The canonical Mooncake FAST'25 fields are ``timestamp`` (relative milliseconds),
``input_length``, ``output_length``, and ordered 512-token ``hash_ids``.  Codex
telemetry contains a scalar cached-input count rather than real prefix block
hashes, so this converter synthesizes anonymous block identities that reproduce
the observed cacheable-prefix length.  The identities do not claim content
lineage or cache residency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
BLOCK_SIZE_TOKENS = 512


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = convert_codex_calls_to_mooncake(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        compress=args.compress,
        keep_jsonl=args.keep_jsonl,
        compression_level=args.compression_level,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


def convert_codex_calls_to_mooncake(
    input_path: Path,
    output_dir: Path,
    *,
    compress: bool = True,
    keep_jsonl: bool = True,
    compression_level: int = 19,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one Mooncake-compatible row per Codex model call."""

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, mode=0o700)

    trace_path = output_dir / "mooncake_trace.jsonl"
    first_arrival_s: float | None = None
    last_arrival_s: float | None = None
    request_count = 0
    source_input_tokens = 0
    source_cached_tokens = 0
    source_output_tokens = 0
    represented_cached_blocks = 0
    replayable_cached_blocks = 0
    total_blocks = 0
    next_hash_id = 0
    cache_pools: dict[tuple[str, str, str], list[int]] = {}
    model_counts: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8") as source, trace_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{input_path}:{line_number} must be a JSON object")

            arrival_s = _nonnegative_float(record.get("arrival_time_s"), "arrival_time_s")
            input_tokens = _positive_int(record.get("input_tokens"), "input_tokens")
            output_tokens = _positive_int(record.get("output_tokens"), "output_tokens")
            cached_tokens = min(
                _nonnegative_int(record.get("cached_input_tokens"), "cached_input_tokens"),
                input_tokens,
            )
            model = str(record.get("model") or "unknown")
            provider = str(record.get("model_provider") or "unknown")
            rollout_id = str(record.get("rollout_id") or "unknown")

            if first_arrival_s is None:
                first_arrival_s = arrival_s
            if last_arrival_s is not None and arrival_s < last_arrival_s:
                raise ValueError(
                    f"{input_path}:{line_number} arrival_time_s is not monotonic"
                )
            last_arrival_s = arrival_s

            block_count = _ceil_div(input_tokens, BLOCK_SIZE_TOKENS)
            cached_block_count = min(
                cached_tokens // BLOCK_SIZE_TOKENS,
                block_count,
            )
            cache_key = (provider, model, rollout_id)
            cache_pool = cache_pools.setdefault(cache_key, [])
            reusable_block_count = min(cached_block_count, len(cache_pool))
            hash_ids = list(cache_pool[:reusable_block_count])
            new_block_count = block_count - reusable_block_count
            hash_ids.extend(range(next_hash_id, next_hash_id + new_block_count))
            next_hash_id += new_block_count
            if len(cache_pool) < block_count:
                cache_pool.extend(hash_ids[len(cache_pool) : block_count])

            timestamp_ms = round((arrival_s - first_arrival_s) * 1000.0)
            mooncake_row = {
                "timestamp": timestamp_ms,
                "input_length": input_tokens,
                "output_length": output_tokens,
                "hash_ids": hash_ids,
            }
            target.write(
                json.dumps(mooncake_row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )

            request_count += 1
            source_input_tokens += input_tokens
            source_cached_tokens += cached_tokens
            source_output_tokens += output_tokens
            represented_cached_blocks += cached_block_count
            replayable_cached_blocks += reusable_block_count
            total_blocks += block_count
            model_counts[model] += 1
            provider_counts[provider] += 1

    if request_count == 0:
        raise ValueError(f"No Codex model-call rows loaded from {input_path}")
    trace_path.chmod(0o600)

    compressed_path: Path | None = None
    if compress:
        compressed_path = output_dir / "mooncake_trace.jsonl.zst"
        _compress_zstd(trace_path, compressed_path, compression_level)
        if not keep_jsonl:
            trace_path.unlink()

    represented_cached_tokens = represented_cached_blocks * BLOCK_SIZE_TOKENS
    represented_cached_tokens = min(represented_cached_tokens, source_input_tokens)
    manifest = {
        "schema_version": "codex-to-mooncake-trace-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": _sha256(input_path),
        },
        "output": {
            "trace": trace_path.name if trace_path.exists() else None,
            "trace_bytes": trace_path.stat().st_size if trace_path.exists() else None,
            "compressed_trace": compressed_path.name if compressed_path else None,
            "compressed_trace_bytes": (
                compressed_path.stat().st_size if compressed_path else None
            ),
            "block_size_tokens": BLOCK_SIZE_TOKENS,
        },
        "semantics": {
            "timestamp": "source arrival_time_s rebased to zero and rounded to ms",
            "input_length": "provider-reported full model input tokens",
            "output_length": "provider-reported model output tokens",
            "hash_ids": (
                "synthetic anonymous block identities; the first "
                "floor(cached_input_tokens/512) blocks reuse a rollout-scoped cache pool "
                "when already observed; remaining blocks are unique and extend that pool"
            ),
            "limitations": [
                "hash_ids reproduce aggregate full-block cacheability, not real content hashes",
                "sub-512-token cached tails are not representable in Mooncake hash_ids",
                "hash identity does not prove cache residency or session lineage",
                "provider token_count timestamps are used as request-arrival proxies",
            ],
        },
        "counts": {
            "requests": request_count,
            "input_tokens": source_input_tokens,
            "cached_input_tokens": source_cached_tokens,
            "output_tokens": source_output_tokens,
            "total_blocks": total_blocks,
            "represented_cached_blocks": represented_cached_blocks,
            "replayable_cached_blocks": replayable_cached_blocks,
            "represented_cached_tokens": represented_cached_tokens,
            "cached_token_representation_ratio": (
                represented_cached_tokens / source_cached_tokens
                if source_cached_tokens
                else 1.0
            ),
            "span_ms": round(
                ((last_arrival_s or 0.0) - (first_arrival_s or 0.0)) * 1000.0
            ),
            "unique_hash_ids": next_hash_id,
            "replayable_cached_block_ratio": (
                replayable_cached_blocks / represented_cached_blocks
                if represented_cached_blocks
                else 1.0
            ),
        },
        "breakdown": {
            "models": dict(sorted(model_counts.items())),
            "providers": dict(sorted(provider_counts.items())),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_checksums(output_dir)
    _make_private(output_dir)
    return manifest


def _compress_zstd(source: Path, target: Path, level: int) -> None:
    executable = shutil.which("zstd")
    if executable is None:
        raise RuntimeError("zstd executable is required for --compress")
    subprocess.run(
        [executable, "-q", f"-{level}", "-T0", "-f", str(source), "-o", str(target)],
        check=True,
    )
    target.chmod(0o600)


def _positive_int(value: Any, field_name: str) -> int:
    parsed = _nonnegative_int(value, field_name)
    if parsed <= 0:
        raise ValueError(f"Codex field must be positive: {field_name}")
    return parsed


def _nonnegative_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Codex row has invalid integer field: {field_name}") from error
    if parsed < 0:
        raise ValueError(f"Codex field must be nonnegative: {field_name}")
    return parsed


def _nonnegative_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Codex row has invalid float field: {field_name}") from error
    if parsed < 0:
        raise ValueError(f"Codex field must be nonnegative: {field_name}")
    return parsed


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _write_checksums(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    target = output_dir / "SHA256SUMS"
    target.write_text(
        "\n".join(f"{_sha256(path)}  {path.name}" for path in paths) + "\n",
        encoding="utf-8",
    )
    target.chmod(0o600)


def _make_private(output_dir: Path) -> None:
    output_dir.chmod(0o700)
    for path in output_dir.iterdir():
        path.chmod(0o600)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write mooncake_trace.jsonl.zst (default: enabled).",
    )
    parser.add_argument(
        "--keep-jsonl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the uncompressed JSONL after creating .zst (default: false; "
            "decompress the archive before Mooncake replay)."
        ),
    )
    parser.add_argument("--compression-level", type=int, default=19)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.compression_level <= 19:
        parser.error("--compression-level must be between 1 and 19")
    if args.keep_jsonl and not args.compress:
        # With compression disabled the JSONL necessarily remains; normalize
        # the otherwise redundant flag rather than treating it as an error.
        args.keep_jsonl = True
    return args


if __name__ == "__main__":
    raise SystemExit(main())
