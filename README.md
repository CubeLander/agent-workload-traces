# Agent Workload Traces

Private, content-free request-shape snapshots for reproducible agent workload
replay. The compact compressed snapshot is tracked directly in this repository.

## Snapshot 2026-08-13

`codex-agent-workload-mooncake-20260813.tar.zst` contains Mooncake-compatible
traces for all recorded Codex subagents, a DeepSeek V4 Flash (Vera) subset, and
a point-in-time GPT-5.6 family subset covering Sol, Luna, and Terra. It preserves
relative timing, input/output lengths, and anonymous rollout-local 512-token
prefix-block reuse. It contains no prompts, responses, reasoning, or tool
content.

See `SNAPSHOT-2026-08-13.json` for counts, hashes, and limitations. Verify the
tracked archive against `SHA256SUMS` before extracting it:

```bash
sha256sum -c SHA256SUMS
```

## Rebuild or extend a snapshot

The repository also owns the temporary conversion tools used to derive these
content-free traces from private local Codex rollout records:

```bash
python3 prepare_codex_subagent_trace.py --selection gpt-5.6 \
  --output /private/codex-agent-workload
python3 convert_codex_calls_to_mooncake_trace.py \
  --input /private/codex-agent-workload/model_calls.jsonl \
  --output /private/codex-agent-mooncake
```

See [`docs/codex-subagent-dataset.md`](docs/codex-subagent-dataset.md) for the
selection, schema, approximation, and privacy contracts. Verbatim rollout
snapshots are private source material and must never be committed here.
