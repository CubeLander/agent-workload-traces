# Agent Workload Traces

Private, content-free request-shape snapshots for reproducible agent workload
replay. Snapshot assets are published as GitHub Releases rather than regular Git
blobs.

## Snapshot 2026-08-13

The `snapshot-2026-08-13` release contains Mooncake-compatible traces for all
recorded Codex subagents and a DeepSeek V4 Flash (Vera) subset. It preserves
relative timing, input/output lengths, and anonymous rollout-local 512-token
prefix-block reuse. It contains no prompts, responses, reasoning, or tool
content.

See `SNAPSHOT-2026-08-13.json` for counts, hashes, and limitations. Verify the
release asset against `RELEASE-SHA256SUMS` before extracting it.
