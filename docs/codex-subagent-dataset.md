# Codex Subagent Workload Dataset

This local dataset captures real Codex subagent workloads from the machine's
rollout JSONL files.  Its primary selection is the worker role informally named
**Vera**: DeepSeek V4 Flash subagents used for research, evidence gathering, and
bounded engineering work.

## Stable Vera identity

The runtime nickname is not an identity: names such as `Hume`, `Planck`, or
`Dirac` change per spawn.  Select Vera sessions using both recorded fields:

```text
session_meta.payload.agent_role == "deepseek_worker"
turn_context.payload.model ends with "deepseek-v4-flash"
```

The suffix rule intentionally accepts the provider-qualified model names seen
on this machine, including `combo/`, `deepseek-direct/`, and `opencode-go/`.

## Build a private snapshot

```bash
python3 prepare_codex_subagent_trace.py \
  --selection vera \
  --include-raw \
  --output /data/private/codex-vera-agent-workload-YYYYMMDD
```

The exporter refuses to replace an existing snapshot unless `--overwrite` is
explicit.  It sets directories to mode `0700` and files to `0600`.

Use `--selection all-subagents` to inventory every recorded subagent role.  The
Vera-only selection is preferable for a DeepSeek V4 workload because it avoids
mixing provider- and model-specific token accounting without a label-aware
analysis.

Use `--selection gpt-5.6` for a point-in-time slice of every subagent whose
recorded model is the GPT-5.6 base model or a family variant such as
`gpt-5.6-sol`, `gpt-5.6-luna`, or `gpt-5.6-terra`. Provider-qualified forms are
accepted. Each live rollout is copied once before parsing, so concurrent appends
after that copy belong to a future snapshot rather than this one.

## Files and semantics

| File | Meaning |
| --- | --- |
| `manifest.json` | Selection predicate, schema, counts, token totals, privacy notes |
| `sessions.jsonl` | One row per selected rollout, with model, role, time range, and source digest |
| `tasks.jsonl` | New subagent work (`trigger_turn=true`), including assignment and terminal handoff when recorded |
| `steering.jsonl` | Follow-up messages delivered to an already-running agent (`trigger_turn=false`) |
| `model_calls.jsonl` | Deduplicated provider token-usage observations with input, cached input, uncached input, output, reasoning output, and timing |
| `raw_rollouts/` | Optional verbatim snapshots for lossless reconstruction |
| `SHA256SUMS` | Integrity receipt for every exported file except itself |

`tasks.jsonl` is the semantic agent workload.  `model_calls.jsonl` is the
traffic-shape workload.  They must not be conflated: the assignment text is
only one incremental input to a long-lived agent context, while
`input_tokens` measures the full provider request context.

The traffic rows therefore do **not** contain fabricated prompts.  A later
load generator should synthesize or reconstruct contexts while preserving:

- `input_tokens`;
- `cached_input_tokens` and `uncached_input_tokens`;
- `output_tokens` and `reasoning_output_tokens`;
- per-session ordering and timing; and
- model/provider labels.

## Content-free Mooncake trace

When only request traffic shape matters, convert `model_calls.jsonl` into the
Mooncake FAST'25 JSONL contract:

```bash
python3 convert_codex_calls_to_mooncake_trace.py \
  --input /private/codex-vera-agent-workload/model_calls.jsonl \
  --output /private/codex-vera-mooncake-trace
```

Each row contains only relative arrival milliseconds, input/output token
lengths, and anonymous 512-token `hash_ids`. The exporter constructs a distinct
cache pool for each agent rollout so repeated leading blocks preserve session
locality without exposing session IDs in the Mooncake trace. The target reuse
length is `floor(cached_input_tokens / 512)`; the manifest separately records
how much was realizable from blocks already observed earlier in that rollout.

The generated hashes are synthetic. They do not reconstruct real content
lineage, session identity, eviction state, or sub-block cached tails. The
manifest records this approximation and its represented cached-token ratio.
The ordinary JSONL is directly compatible with Mooncake's trace parser. By
default only the compact `.zst` is retained; decompress it before replay:

```bash
zstd -d mooncake_trace.jsonl.zst -o mooncake_trace.jsonl
```

Pass `--keep-jsonl` when immediate replay is more important than storage.

## Privacy boundary

Curated task text redacts obvious private keys, authorization headers,
credential assignments, and credential-bearing URLs.  Pattern matching is not
a proof that text is secret-free.

With `--include-raw`, `raw_rollouts/` is deliberately verbatim.  It can contain
system instructions, user content, source excerpts, shell commands and output,
host paths, internal URLs, or secrets.  Keep the snapshot private, do not add
it to Git, and only derive shareable subsets after a separate content review.

This repository ignores generated `*.jsonl` and `data/processed/*`, so the default local output stays
outside normal Git state.  For durable storage, prefer a private data volume
and retain `manifest.json` plus `SHA256SUMS` with the protected artifact.
