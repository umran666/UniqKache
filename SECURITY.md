# Security Policy

## Scope

UniqKache is a **research prototype** for KV-cache management. It is not
intended to serve untrusted users, and it is not hardened for production
deployment. The following are explicitly **out of scope**:

- Denial of service via adversarial inputs to a served model.
- Prompt injection, jailbreaks, or model-level safety behaviour.
- Multi-tenant isolation of cache memory.

If you are deploying this in a context where those matter, you are outside the
project's supported envelope.

## In scope

We do care about the following, because they are bugs in *our* code:

- **Memory-safety defects in cache management** — out-of-bounds indexing,
  silent truncation, use of freed/evicted state, or shape corruption in the
  K/V store that could produce wrong results without an error.
- **Silent correctness failures** — a policy that reports a memory saving while
  returning incorrect attention results without raising or logging.
- **Arbitrary code execution via deserialisation** — loading a malicious
  experiment config or checkpoint. Configs are parsed as JSON/YAML data and are
  never `eval`'d; report any path that violates this.
- **Path traversal** in benchmark/experiment result writers.
- **Credential leakage** — e.g. a HF token being written into a committed
  result record or log file.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Use GitHub's private vulnerability reporting:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Include: affected version/commit, a minimal reproduction, and the impact.

If private reporting is unavailable, open a public issue containing only the
words *"security report — requesting private channel"* and no technical detail.

## Response expectations

This is a volunteer research project. We aim to acknowledge reports within
**7 days**. There is no bug bounty. We will credit reporters in the release
notes unless asked not to.

## Supported versions

Only the `main` branch is supported. There are no maintained release branches
and no backports while the project is pre-1.0.
