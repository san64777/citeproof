## What and why

<!-- What does this change, and why? Link any related issue. -->

## Checklist

- [ ] Tests added or updated (the binder is built test-first)
- [ ] `uv run ruff check src tests` is clean
- [ ] `uv run mypy src` is clean
- [ ] `uv run pytest` is green
- [ ] `uv run pip-licenses --fail-on='GPL;AGPL;LGPL;SSPL' --partial-match` passes
- [ ] Conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`)
- [ ] No em or en dashes (use a hyphen, a comma, or restructure)
- [ ] For a change to verify / decompose / cite-gate logic: it never cites a non-`OK` page, no model
      judges its own output, and it prefers abstaining to a confident wrong citation. A fixture locks
      the behavior in.
