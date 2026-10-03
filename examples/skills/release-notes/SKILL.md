---
name: release-notes
description: Write release notes from the git history and the diff since the last tag.
---
# Release notes

1. Run `git_status` and `git_diff` to see uncommitted work; mention it separately.
2. Read the commit history with `run_command(argv=["git", "log", "--oneline", "-30"])`.
3. Group changes into **Added**, **Changed**, **Fixed**, and **Removed**.
4. One line per change, written for users, not for developers: what changed and why
   it matters. Leave out refactors that change no behaviour.
5. End with upgrade notes when a config key, CLI flag, or API field changed.
