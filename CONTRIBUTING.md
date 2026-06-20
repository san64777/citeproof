# Contributing to citeproof

Thanks for your interest. citeproof is a local research agent that won't cite a source it can't
verify is real: every claim links to a verified-OK page and a highlighted supporting line.
Contributions are welcome. Please read this first so your PR goes smoothly.

citeproof is **pre-M0** (see the README): the binder must clear a hard, pre-registered accuracy gate
before any UI is built. Until it does, the most valuable contributions are adversarial, not features.

## Project philosophy (please respect it)

- **Abstain over guess.** A cited claim whose receipt does not actually support it is the one failure
  this project exists to prevent. Dropping a claim it cannot verify is a correct answer, not a bug. A
  false citation (attaching a receipt to a claim the source never made) is the cardinal sin, worse
  than a miss.
- **Verification is the moat.** When a source is behind a hard wall, citeproof excludes it and says
  so. Do not frame the project as a way around anti-bot systems; the edge is that every citation is
  verifiable.
- **Two independent signals.** The binder pairs a neural entailment check with an orthogonal symbolic
  check (numbers, dates, quantifiers, negation, direction). They must stay genuinely independent: a
  second same-recipe model has correlated errors and buys little.
- **No model audits itself.** The check that a decomposition is faithful must not be the same model
  that later verifies the claim. Circularity (a model blessing its own output) is rejected at runtime.
- **Cite only strict `OK`.** Only a page that veriscrape verdicts `OK` can be a source. `UNVERIFIED`
  is excluded exactly like `BLOCKED`. Do not loosen this.
- **Permissive only.** The runtime dependency tree must stay permissive (MIT / BSD / Apache / MPL). No
  GPL / AGPL / LGPL / SSPL, even transitively. CI enforces this from the first commit. (SingleFile is
  AGPL and is used as a subprocess only, never imported, so it never enters the runtime tree.)
- **Gates before code.** Nothing past the M0 binder gate gets built until M0 passes. Please do not
  open feature PRs for the UI or later milestones yet.

## Development setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<your-fork>/citeproof
cd citeproof
uv sync
uv run pytest
```

The deterministic core has no heavy-model dependency, so the default test suite runs with no models
and no network. The real models (the entailment verifier, the embedder, the decomposer, the
round-trip arbiter) live behind an optional `binder` extra and lazy adapters; opt-in integration tests
run only when you install the extra and set `CITEPROOF_RUN_MODEL_TESTS=1`.

## Before you open a PR

Run the full quality bar locally. All of it must be green:

```bash
uv run ruff check src tests     # lint (line length 110)
uv run mypy src                 # types
uv run pytest                   # tests (deterministic core; models are opt-in)
uv run pip-licenses --fail-on='GPL;AGPL;LGPL;SSPL' --partial-match   # license gate
```

- **Add tests** for any behavior change. The binder is built test-first.
- **A change to the verify, decompose, or cite-gate logic** must keep the gate honest: never cite a
  non-`OK` page, never let a model judge its own output, and prefer abstaining to a confident wrong
  citation. Add a fixture that locks the behavior in.
- **Conventional commit messages** (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- **No em or en dashes** in code, comments, or docs. Use a hyphen, a comma, or restructure.

## The most useful contributions right now

- **Hard claim/source pairs** for the eval set: clean entailed, clean not-entailed, near-miss
  paraphrases (a flipped number or a swapped entity), and decontextualization traps.
- **Red-teaming the gate's honesty:** a way to make the binder cite a claim its source does not
  support, or to make a junk page slip the verdict gate.

## Workflow

1. **Fork** and create a branch (`git checkout -b feat/my-change`).
2. Make your change with tests.
3. Run the quality bar above.
4. Open a **Pull Request** against `main`. CI (lint, types, tests, license gate) runs automatically.
5. A maintainer reviews and merges. Thanks!

## License

By contributing, you agree your contributions are licensed under the project's
[Apache-2.0](LICENSE) license.
