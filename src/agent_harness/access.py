"""Identity, roles, tenants, and quotas for the HTTP API (and the CLI's quota checks).

API keys map to principals: a name, a tenant, and a role. Keys are configured
by their SHA-256 (``key_sha256``) or read from an environment variable
(``key_env``), so config never holds a usable secret. Roles are ordered:

* ``viewer``   - read runs, events, usage, approvals, profiles, templates
* ``operator`` - viewer + create, continue, steer, cancel, resume, approve
* ``admin``    - operator + change permission mode and rewind

A principal only sees runs of its own tenant (``*`` = every tenant).
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import IntEnum
from typing import Any

from agent_harness.domain import utc_now


class Role(IntEnum):
    VIEWER = 1
    OPERATOR = 2
    ADMIN = 3

    @classmethod
    def parse(cls, value: str) -> "Role":
        try:
            return cls[value.upper()]
        except KeyError as exc:
            raise ValueError(f"role must be viewer, operator, or admin (got {value!r})") from exc


@dataclass(frozen=True, slots=True)
class Principal:
    name: str
    tenant: str
    role: Role

    def can_access(self, tenant: str) -> bool:
        return self.tenant == "*" or self.tenant == tenant

    @property
    def creation_tenant(self) -> str:
        return "default" if self.tenant == "*" else self.tenant


@dataclass(frozen=True, slots=True)
class ApiKey:
    name: str
    tenant: str
    role: Role
    key_sha256: str


def sha256_hex(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class PrincipalDirectory:
    """Resolve a bearer token to a principal (constant-time comparison of hashes)."""

    def __init__(self, keys: Sequence[ApiKey], *, legacy_token: str | None = None) -> None:
        self._keys = tuple(keys)
        self._legacy = sha256_hex(legacy_token) if legacy_token else None

    @property
    def enabled(self) -> bool:
        return bool(self._keys) or self._legacy is not None

    def resolve(self, token: str) -> Principal | None:
        digest = sha256_hex(token)
        found: Principal | None = None
        for key in self._keys:
            if hmac.compare_digest(digest, key.key_sha256):
                found = Principal(key.name, key.tenant, key.role)
        if found is None and self._legacy is not None and hmac.compare_digest(digest, self._legacy):
            found = Principal("api-token", "default", Role.ADMIN)
        return found


def api_keys_from_config(entries: Sequence[Mapping[str, Any]], environ: Mapping[str, str] | None = None) -> list[ApiKey]:
    env = environ if environ is not None else os.environ
    keys: list[ApiKey] = []
    for entry in entries:
        digest = str(entry.get("key_sha256") or "")
        if not digest and entry.get("key_env"):
            secret = env.get(str(entry["key_env"]), "")
            if len(secret) < 16:
                raise ValueError(f"api key {entry.get('name')!r}: {entry['key_env']} must hold at least 16 characters")
            digest = sha256_hex(secret)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"api key {entry.get('name')!r}: key_sha256 must be a lowercase sha256 hex digest")
        keys.append(ApiKey(str(entry["name"]), str(entry.get("tenant", "default")),
                           Role.parse(str(entry.get("role", "viewer"))), digest))
    return keys


# -- quotas --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Quota:
    tenant: str
    max_runs_per_day: int | None = None
    max_tokens_per_day: int | None = None
    max_cost_usd_per_day: float | None = None
    max_concurrent_runs: int | None = None


class QuotaExceeded(Exception):
    """A tenant is over one of its limits."""


async def check_quota(store: Any, quotas: Mapping[str, Quota], tenant: str, *, new_run: bool = True) -> None:
    """Raise ``QuotaExceeded`` when ``tenant`` may not start (or continue) work now.

    Usage is a rolling 24-hour window from the run store. Stores without
    ``tenant_usage`` (custom stores) are not limited.
    """

    quota = quotas.get(tenant) or quotas.get("*")
    usage_of = getattr(store, "tenant_usage", None)
    if quota is None or usage_of is None:
        return
    usage = await usage_of(tenant, utc_now() - timedelta(days=1))
    if new_run and quota.max_runs_per_day is not None and usage.runs >= quota.max_runs_per_day:
        raise QuotaExceeded(f"tenant {tenant!r} reached its limit of {quota.max_runs_per_day} runs per day")
    if new_run and quota.max_concurrent_runs is not None and usage.active_runs >= quota.max_concurrent_runs:
        raise QuotaExceeded(f"tenant {tenant!r} already has {usage.active_runs} active runs "
                            f"(limit {quota.max_concurrent_runs})")
    if quota.max_tokens_per_day is not None and usage.tokens >= quota.max_tokens_per_day:
        raise QuotaExceeded(f"tenant {tenant!r} used its {quota.max_tokens_per_day} tokens for the day")
    if quota.max_cost_usd_per_day is not None and usage.cost_usd >= quota.max_cost_usd_per_day:
        raise QuotaExceeded(f"tenant {tenant!r} spent its ${quota.max_cost_usd_per_day:.2f} for the day")
