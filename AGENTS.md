# Agent Workload Traces Guidance

This private repository owns content-free, reproducible request-shape snapshots
and the small exporters that produce them. Keep prompts, responses, reasoning,
tool content, verbatim rollout JSONL, credentials, private URLs, and host-local
Codex state out of Git even though the repository is private.

`prepare_codex_subagent_trace.py` may create a verbatim `raw_rollouts/` tree only
under an explicitly private output directory. Never commit that output. Curated
text redaction is a safety layer, not proof that arbitrary text is publishable.
The tracked Mooncake trace contains only relative timing, token lengths, and
anonymous synthetic prefix-block identities.

Before publication run:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q *.py tests
sha256sum -c SHA256SUMS
```
