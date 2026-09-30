# Deploying agent-harness

This guide runs the HTTP API and queue workers against PostgreSQL. `deploy/` has a
working compose setup: Postgres, the API, and two workers.

## Pieces

| Component | Command | Scales |
|---|---|---|
| API | `agent-harness serve --config models.toml --host 0.0.0.0 --workers 0` | Horizontally, behind a load balancer |
| Workers | `agent-harness worker --config models.toml --workers N` | Horizontally. Runs are claimed with `FOR UPDATE SKIP LOCKED` leases |
| Database | PostgreSQL 14 or newer (`[postgres]` extra) | One primary. Take backups |

- **Which store:** the API and the workers share only the database. A single-host setup
  can keep SQLite (`--database runs.sqlite3`); anything with several hosts needs
  Postgres.
- **Workers inside the API:** `serve --workers N` also runs N workers in the API
  process.

## Quick start with compose

```bash
cp deploy/.env.example deploy/.env      # set POSTGRES_PASSWORD and the API keys
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up --build
curl -s localhost:8000/readyz
curl -s -H "Authorization: Bearer $AGENT_HARNESS_TEAM_A_KEY" \
     -d '{"prompt": "Summarize the repository"}' localhost:8000/runs
```

Edit `deploy/models.toml` first so its model profile points at your model server.

## Configuration checklist

- **Storage.** Set `[storage] url_env = "AGENT_HARNESS_DATABASE_URL"` (or pass
  `--database postgresql://...`).
  - Schema migrations run on startup, one migrator at a time, and are recorded in
    `schema_migrations`.
  - A database newer than the code is refused.
- **API keys.** Each key is a principal with a tenant and a role:
  - `viewer` reads.
  - `operator` creates, continues, steers, cancels, resumes, and approves runs.
  - `admin` can also change modes and rewind.

  ```toml
  [[api.keys]]
  name = "team-a"
  tenant = "team-a"
  role = "operator"
  key_env = "AGENT_HARNESS_TEAM_A_KEY"   # or key_sha256 = "<sha256 of the key>"
  ```

  A key only sees its own tenant's runs; others return 404. `tenant = "*"` sees every
  tenant. Approvals and mode changes record the acting key's name. The older
  `AGENT_HARNESS_API_TOKEN` still works as an admin of tenant `default`.
- **Quotas.** `[[quotas]] tenant, max_runs_per_day, max_tokens_per_day,
  max_cost_usd_per_day, max_concurrent_runs`.
  - The window is a rolling 24 hours, and `tenant = "*"` is the default for every tenant.
  - Quotas are checked when a run is created or continued (the API returns HTTP 429).
  - They are not re-checked mid-run, so combine them with per-run budgets.
- **Secrets.** Credential fields accept an environment variable name, `env:NAME`, or
  `file:/run/secrets/name`. The file form works with Docker and Kubernetes secret mounts.
  Error messages name the reference, never the value.
- **Egress.** With `[network] restrict_egress = true`, HTTP tools, web search, and MCP
  over HTTP may only reach `allowed_hosts`. `*.example.com` also matches `example.com`.
  Model endpoints are not affected.
- **Redaction.** Redaction is on by default.
  - Bearer tokens, `sk-`/`tvly-`/`ghp_` keys, AWS keys, private keys, and
    `password=`/`token=` values are masked in persisted events, run errors, and telemetry.
  - Add your own with `[redaction] patterns = ["..."]`.
  - The conversation the model resumes from is not rewritten, so protect the database
    itself.
- **Encryption at rest.** This is not done by the harness. Use an encrypted volume or
  your Postgres provider's storage encryption.
- **Retention.** `agent-harness gc --older-than-days 30` lists finished runs older than
  the cutoff, and `--yes` deletes them with their events and queue, steering, and
  idempotency rows. Artifacts are content-addressed and may be shared, so clean
  `artifacts/` separately.
- **Coding mode in containers.** Its Docker sandbox needs a Docker daemon, which the API
  and worker containers do not have. Run coding workloads on hosts with Docker, or use
  `execution_runtime = "local"` only inside an already isolated container.

## Health

- **Liveness:** `GET /healthz` answers while the process is alive.
- **Readiness:** `GET /readyz` checks that the store is reachable and reports the schema
  version. It returns HTTP 503 when the store is not reachable.

## Upgrades

1. Back up the database.
2. Deploy the new image. The first process to start applies any new migrations inside a
   transaction; the others wait on an advisory lock.
3. Roll workers gradually. A run left mid-step by a stopped worker is picked up again
   after its lease expires, and resumed from its last checkpoint.

## Verifying a deployment

```bash
AGENT_HARNESS_TEST_POSTGRES_DSN=postgresql://user:pass@host/db_for_tests \
  python -m unittest tests.test_store_conformance -v   # uses a disposable database
agent-harness doctor --config models.toml --model-profile <profile>   # live model check
```
