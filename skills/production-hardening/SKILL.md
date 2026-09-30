---
name: production-hardening
description: Production concerns — versioned schema migrations, the RunStore conformance suite, the optional Postgres store, API principals/tenants/roles, per-tenant quotas, secret references, network egress allowlists, redaction, retention, and deployment (container, compose, health/readiness). Use for any Phase 9 item.
---

# Production Hardening

## Storage (9a)

- **Migrations:** `persistence/migrations.py` keeps an ordered list of `(version,
  statements)`.
  - SQLite tracks the version in `PRAGMA user_version`; Postgres uses a
    `schema_migrations` table.
  - Each migration runs in one transaction on open. A database newer than the code is
    refused with a clear error.
  - Existing databases (created by `CREATE TABLE IF NOT EXISTS`) are adopted as
    version 1.
- **Conformance:** `tests/store_conformance.py` is a mixin that every store must pass:
  runs, checkpoints and optimistic concurrency, events, the queue and its leases,
  steering, idempotency, and the tool catalog. It runs against the memory and SQLite
  stores always, and against Postgres only when `AGENT_HARNESS_TEST_POSTGRES_DSN` is set
  (otherwise skipped).
- **Postgres:** `PostgresRunStore`, behind the `[postgres]` extra (`psycopg` 3, imported
  lazily).
  - It is selected with `--database postgresql://…` or `[storage] url`.
  - It uses the same tables as SQLite, with `jsonb` payloads, `FOR UPDATE SKIP LOCKED`
    for queue claims, and a connection per operation from a small pool.
  - Long-term memory stays on SQLite FTS; it is documented as a single-host feature.

## Identity, tenancy, quotas (9b)

- **API principals:** `[[api.keys]] name, tenant, role, key_sha256`. The API stores only
  the SHA-256 of each key; the keys themselves come from an environment variable or a
  secret file.
  - Roles: `viewer` (read), `operator` (create, continue, steer, cancel, approve), and
    `admin` (everything, including tools and mode).
  - The legacy single `--api-token` still works and acts as an admin with tenant
    `default`.
- **Run ownership:** runs carry `tenant` and `created_by` (fields with defaults). The API
  refuses access across tenants, and approvals and mode changes record the acting
  principal.
- **Quotas:** `[[quotas]] tenant, max_runs_per_day, max_tokens_per_day,
  max_cost_usd_per_day, max_concurrent_runs`.
  - They are checked when a run is created and when one is continued (API: HTTP 429;
    CLI: exit code 2) over a rolling 24-hour window.
  - Usage is aggregated from indexed run columns (tenant, status, created_at, tokens,
    cost), not from JSON.
  - They are not re-checked on worker claim or in the middle of a run: per-run budgets
    bound in-flight work.

## Secrets and network (9b)

- **Secret references:** `SecretResolver` resolves `env:NAME` and `file:/path`
  references. Every `*_env` field keeps working.
- **Egress:** `[network] restrict_egress = true` plus `allowed_hosts = ["api.tavily.com",
  "*.example.com"]` applies to HTTP tools, web search, and MCP over HTTP.
  - With `restrict_egress` and an empty list, tools make no outbound HTTP.
  - Model endpoints are not affected.
  - A violation is a tool error naming the host.

## Data governance (9c)

- **Redaction:** `[redaction]` holds built-in patterns (bearer tokens, `sk-…`/`tvly-…`
  style keys, AWS keys, private-key blocks) plus user regexes.
  - It is applied to event payloads, run errors, and telemetry attributes by wrapping the
    store the orchestrator writes through (`RedactingRunStore`).
  - The conversation snapshot the model resumes from is not rewritten: the model needs
    its own context, so protect the database at rest.
- **Retention:** `agent-harness gc --older-than-days DAYS [--yes]` purges terminal runs
  older than the cutoff: their events, queue, steering, and idempotency rows.
  - Without `--yes` it is a dry run.
  - Events stay append-only except inside the purge transaction: a guard row in SQLite, a
    `SET LOCAL` flag in Postgres.
  - Artifacts are content-addressed and may be shared, so they are left for a separate
    cleanup.
- **Encryption at rest:** not implemented in code. Use volume encryption, or Postgres
  with encrypted storage. This is documented, not faked.

## Deployment (9d)

- **Container:** a `Dockerfile` with a non-root user, an entrypoint of `agent-harness`,
  and no secrets baked in.
- **Compose:** `deploy/docker-compose.yml` runs the API, two workers, and Postgres, with
  health checks.
- **Health:** `/healthz` reports liveness. `/readyz` checks that the store is reachable
  and the migrations are current.
- **Docs:** `docs/DEPLOYMENT.md` covers configuration, secrets, scaling workers,
  backups, and upgrades.

## Testing

Everything above stays network-free:
- **Postgres:** gated behind its DSN environment variable.
- **Egress, redaction, quotas, roles:** exercised with fakes and in-memory stores.
- **Deployment:** the Dockerfile and compose file are validated only by `docker build`
  and `docker compose config` when Docker is present. That is a manual check, not a unit
  test.
