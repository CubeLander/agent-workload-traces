# Agent Workload Traces

Private, content-free request-shape snapshots for reproducible agent workload
replay. The compact compressed snapshot is tracked directly in this repository.

## Snapshot 2026-08-13

`codex-agent-workload-mooncake-20260813.tar.zst` contains Mooncake-compatible
traces for all recorded Codex subagents and a DeepSeek V4 Flash (Vera) subset.
It preserves relative timing, input/output lengths, and anonymous rollout-local
512-token prefix-block reuse. It contains no prompts, responses, reasoning, or
tool content.

See `SNAPSHOT-2026-08-13.json` for counts, hashes, and limitations. Verify the
tracked archive against `SHA256SUMS` before extracting it:

```bash
sha256sum -c SHA256SUMS
```
