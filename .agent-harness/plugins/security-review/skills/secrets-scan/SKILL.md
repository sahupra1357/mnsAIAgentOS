---
name: secrets-scan
description: Find hard-coded secrets (API keys, tokens, passwords, private keys) in the workspace without revealing them.
---
# Secrets scan

1. Search with `search_files` for these patterns, one at a time:
   `api[_-]?key`, `secret`, `token`, `password`, `BEGIN .*PRIVATE KEY`,
   `sk-[A-Za-z0-9]`, `tvly-`, `ghp_`, `AKIA[0-9A-Z]{16}`.
2. Skip `.env.example`, test fixtures with obvious placeholders (`replace-me`,
   `xxx`, `dummy`), and lines that only read an environment variable.
3. For each real hit, read a few lines around it with `read_file` to confirm it is a
   literal value, not a variable name or a lookup.
4. Report file and line only. Never copy the secret's value into your answer.
5. For each finding, recommend moving the value to `.env` (gitignored) and reading it
   by name, and rotating the secret if it was ever committed.
