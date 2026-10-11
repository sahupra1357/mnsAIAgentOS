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
  - **Admission:** creating or continuing a run reserves budget atomically and caps the
    run's budget to what was granted (HTTP 429 when nothing is left).
  - **Step checks:** the orchestrator re-checks the tenant at every step (every
    `quota_check_seconds`).
  - **Overshoot:** at most one model call's tokens, since usage is known only after the
    call returns.
- **Per-user quotas.** `[user_quotas] max_runs_per_day, max_tokens_per_day,
  max_cost_usd_per_day, max_concurrent_runs` apply to every end user (the API's `user`
  field) within their tenant; `[user_quotas.users.NAME]` replaces them for one user; 0 means
  no limit. They are reserved in the same transaction as the tenant quota and re-checked at
  each step. Upgrading adds a column and indexes (SQLite migration 7, Postgres migration 4).
- **Rate limits.** `[api] rate_limit_per_minute` and `rate_limit_burst` limit requests per
  API key (per client address without keys): HTTP 429 with `Retry-After`. Each API process
  keeps its own counters, so with N replicas a caller can reach N times the rate; put a
  limit at the load balancer too if that matters.
- **Deadlines for paused runs.** `[interaction] approval_timeout_seconds`,
  `question_timeout_seconds`, and `on_timeout = "deny" | "cancel" | "fail"`. Workers apply
  them at every reconcile; without workers, run `agent-harness expire` from cron.
- **Overrides.** Any table value can be set from the environment, which suits containers:
  `AGENT_HARNESS_SET__API__RATE_LIMIT_PER_MINUTE=120`.
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
- **Memory.** `[memory] backend = "postgres"` keeps long-term memory in Postgres, with the
  URL from `[memory] url_env` or `[storage] url_env`. To move existing notes, run
  `agent-harness memory migrate --from sqlite:memory.sqlite3 --config models.toml`. It can
  be rerun safely, and existing notes are skipped.
- **Docker for sandboxes.** Coding commands and sandboxed Python tools need a Docker daemon
  where workers run, and the API and worker containers from `deploy/` have none.
  - Run workers that use them on hosts with Docker, or mount the Docker socket into the
    worker. The socket gives root-equivalent access to the host, so do this only on
    dedicated hosts.
  - `[coding.docker] mode = "per_run"` keeps one warm container per run; measured locally
    at about 55 ms per command, against about 150 ms for a fresh container per command.
  - Coding images need the tools the agent runs: `python:3.12-slim` has no `git`.

## Modal model servers

The harness calls Modal deployments through the `modal` profile (and `modal_shared` borrows its
embed and guard). Everything is in `deploy/modal/`, one folder per layout, each with its deploy
files and its own scripts:

| Folder | Contents |
|---|---|
| `deploy/modal/scripts/` | one deployment per model, `vllm_server.py` (chat, H200), `vllm_embed_server.py` (L4), `vllm_guard_server.py` (L4), `sglang_server.py`, and their lifecycle scripts `deploy.sh`, `start.sh`, `stop.sh`, `undeploy.sh` (+ `_apps.sh`) |
| `deploy/modal/shared_l4/` | `vllm_companions_server.py` + `router.py` (embeddings and the guard on one L4), and their own `deploy.sh`, `start.sh`, `stop.sh`, `undeploy.sh` (+ `_apps.sh`) |

### Lifecycle scripts

Two independent sets with the same four script names. They take no arguments, never act on each
other's apps, and work from any directory.

Separate group (`deploy/modal/scripts/`: embed L4, guard L4, chat H200; deploy and start go embed,
guard, chat, stop and undeploy go chat, guard, embed):

```bash
deploy/modal/scripts/deploy.sh       # modal deploy, one after the other (no GPU starts)
deploy/modal/scripts/undeploy.sh     # modal app stop, one after the other (asks first)
deploy/modal/scripts/start.sh        # warm up: doctor modal.embed, modal.guard, modal (H200)
deploy/modal/scripts/stop.sh         # modal container stop: billing ends, apps stay deployed

agent-harness doctor --model-profile modal.embed
agent-harness doctor --model-profile modal.guard
agent-harness doctor --model-profile modal --max-tokens 512
```

Shared group (`deploy/modal/shared_l4/`: only `agent-harness-vllm-companions`, embed + guard on one
L4; the H200 is never touched):

```bash
deploy/modal/shared_l4/deploy.sh     # modal deploy (no GPU starts)
deploy/modal/shared_l4/undeploy.sh   # modal app stop (asks first)
deploy/modal/shared_l4/start.sh      # warm up: doctor modal.guard, then modal.embed
deploy/modal/shared_l4/stop.sh       # modal container stop

agent-harness doctor --model-profile modal.guard
agent-harness doctor --model-profile modal.embed
```

#### `.env` drives the Modal warm-up

`doctor` and both `start.sh` scripts do not pick an app: they send a request to the URL that
`.env` sets for the profile, and Modal starts whichever container serves that URL.

| `.env` line | Profile and `doctor` command | Separate group needs | Shared group needs |
|---|---|---|---|
| `AGENT_HARNESS_MODEL__MODAL__ENABLED=true` | the whole `modal` profile (also borrowed by `modal_shared`) | set | set |
| `AGENT_HARNESS_MODEL__MODAL__ENDPOINT` | `modal`: `doctor --model-profile modal` → H200 chat | `…--agent-harness-vllm-serve.modal.run/v1` | not used |
| `AGENT_HARNESS_MODEL__MODAL__EMBEDDING_ENDPOINT` | `modal.embed`: `doctor --model-profile modal.embed` | `…--agent-harness-vllm-embed-serve.modal.run/v1` | `…--agent-harness-vllm-companions-serve.modal.run/v1` |
| `AGENT_HARNESS_MODEL__MODAL__GUARD_ENDPOINT` | `modal.guard`: `doctor --model-profile modal.guard` | `…--agent-harness-vllm-guard-serve.modal.run/v1` | `…--agent-harness-vllm-companions-serve.modal.run/v1` |

- **`.env` drives the warm-up.** `doctor` and both `start.sh` scripts warm whatever URL the line
  for that profile names. So the same `doctor --model-profile modal.guard` warms the separate guard
  or the shared container, depending on `.env`. A commented-out line falls back to the `WORKSPACE`
  placeholder in `models.toml`: nothing is reached, so nothing warms (and nothing bills). For
  example, with `ENDPOINT` commented out the H200 cannot be warmed.
- **`.env` does not affect deploy, undeploy, or stop.** Those scripts use the Modal app names.
- **Switching layouts:** change the two companion lines in `.env`, then restart `serve` (`.env` is
  read when the program starts). `serve` and `/ui` send real requests to the same URLs.

For a single app, use Modal directly: `modal container list` then `modal container stop <id>`, or
`agent-harness doctor --model-profile modal.guard` to warm only the guard.

- **Start** has no Modal command for a web endpoint: a container starts on its first request.
  `start.sh` sends that request through `agent-harness doctor`, which waits through the cold
  start and checks the model, using the repository's `models.toml` and `.env`. A failed start is
  reported and the next app is still started (exit code 1 at the end).
- **Stop** keeps the app and URL. The next request (or `start.sh`) starts it again, with a cold
  start. Requests running on a stopped container are cancelled.
- **Undeploy** removes the app: its URL answers 404 until you deploy again.

### Modal: one L4 for embeddings and the guard

`deploy/modal/shared_l4/vllm_companions_server.py` (app `agent-harness-vllm-companions`) runs
both companion models in one container on one L4:

```
router.py :8000 -- /v1/embeddings  -> embed vLLM :8001  (nomic-embed-text, 5% of the GPU)
                -- /v1/models      -> both lists merged (doctor checks the list)
                -- /health         -> 200 only when both servers are healthy
                -- everything else -> guard vLLM :8002  (Llama Guard 3 8B, 87% of the GPU)
```

How it starts: the router starts the embed server, waits for its `/health`, then starts the
guard server and waits again. Each vLLM process reserves its share of the GPU when it starts, so
starting both at once can make the second fail. Only then does the router open port 8000, and
Modal sends no request before that port answers. If either process exits later, the router stops
the other and exits, and the container ends, so Modal replaces it instead of keeping a half-broken
container warm.

To use it:
1. `deploy/modal/shared_l4/deploy.sh` (same secrets as the guard: `agent-harness-llm`,
   `agent-harness-hf`).
2. In `.env`, point both companion endpoints at it:
   `AGENT_HARNESS_MODEL__MODAL__EMBEDDING_ENDPOINT=https://<workspace>--agent-harness-vllm-companions-serve.modal.run/v1`
   and the same URL for `AGENT_HARNESS_MODEL__MODAL__GUARD_ENDPOINT`.
3. `deploy/modal/shared_l4/start.sh` (doctor for `modal.guard`, then `modal.embed`).
4. Optionally remove the separate pair: `modal app stop agent-harness-vllm-embed` and
   `modal app stop agent-harness-vllm-guard` (`deploy/modal/scripts/undeploy.sh` would also remove the H200 app).

Trade-offs against the separate deployments:

| | Separate (default) | Shared L4 |
|---|---|---|
| GPUs while both are used | 2 x L4 | 1 x L4 |
| Embed cold start | its own, when memory is first used | none once the guard is up |
| Guard KV cache | about 3 GiB (85% of the GPU) | about 2.5 GiB, roughly 20k tokens: `MAX_MODEL_LEN = 8192` and two checks at once; bursts queue (at 80% it was 0.98 GiB and the guard did not start) |
| Start, idle, scale, redeploy | each on its own | together; a bad guard update also takes down embeddings |
| Moving parts | vLLM only | vLLM x 2 + router |

Embeddings are only used when a run calls `memory_search` or `memory_write`, so the saving is
small unless memory is used often. Either way, warm the companions before you use `/ui`
(`agent-harness doctor --model-profile modal.guard`, or `deploy/modal/shared_l4/start.sh`), or the first turn waits for the guard's cold start.

Scaling: Modal scales a deployment by adding containers, each with the same contents. With the
separate layout one container is one vLLM on one GPU, so the guard and embed scale on their own
(`MAX_CONTAINERS`). With the shared layout every added container carries both models on its own
L4, so more guard capacity also brings another embed server.

The first deploy (2026-10-11) used guard 0.80 / embed 0.10 and the guard failed: 15.0 GiB of
weights, 0.35 GiB of CUDA graphs, and about 1.7 GiB of activations left 0.98 GiB of KV cache, under
the 1.0 GiB one 8192-token request needs. The split is now 0.87 / 0.05 (the embed server measured
0.4 GiB). If the guard still fails with too little KV cache memory, lower `MAX_MODEL_LEN`, or set
`GUARD_EXTRA_ARGS = ["--enforce-eager"]` (no CUDA graphs: +0.35 GiB, slightly slower).

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
