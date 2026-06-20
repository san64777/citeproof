# citeproof: provable beats persuasive

Most research assistants compete on the wrong axis. They race to sound smarter: longer answers,
more sources, more confident prose. But a local model, or any model, can write a fluent paragraph
that cites a page which was actually a login wall, or a line that is close to the claim but does not
say it. The output looks authoritative and is quietly wrong. You find out three steps later, in a
report you have already shared.

I do not think "smarter" is the interesting problem. The interesting problem is **provable**: can you
click any sentence and land on the exact line of a real page that supports it? citeproof is built
around that single question, and it gives up a lot to answer it honestly.

## A 200, and a citation, are both untyped results

This is the same idea as [veriscrape](https://github.com/san64777/veriscrape), one layer up.
veriscrape's thesis is that an HTTP `200 OK` is no longer ground truth: it is often a challenge page,
a login wall, a soft-404, or an empty JavaScript shell, and a status-code-only fetcher stores that
husk as data. citeproof's thesis is that a **citation** is no longer ground truth either. A URL next
to a sentence tells you nothing about whether that page is real, whether it is reachable, or whether
it actually contains the claim. You take it on faith. That faith is misplaced often enough to be a
real cost.

veriscrape verifies the fetch. citeproof verifies the claim. Same discipline, two layers.

## Two gates, and a receipt

A claim becomes a citation in citeproof only if it clears two independent gates.

1. **The page is verified real.** citeproof reuses veriscrape as a verdict gate. Only a page that
   verdicts strict `OK`, genuine server-rendered origin content, can be a source. A `BLOCKED` page, a
   `LOGIN_WALL`, a `SOFT_404`, a `HONEYPOT`, an `EMPTY_SHELL`, even an `UNVERIFIED` page where
   veriscrape could not be sure, are all excluded and marked "couldn't verify." citeproof never cites
   what it could not verify.
2. **The claim is entailed by the page.** A local entailment model checks that a verified-OK span
   genuinely supports the claim. It is backed by an orthogonal symbolic check (a flipped number, a
   swapped year, a dropped "not", a flipped quantifier, a reversed direction) that catches the
   near-miss paraphrases a single neural model tends to wave through. The supporting span is anchored
   to a verbatim passage so the receipt can highlight it. If nothing clears both, citeproof abstains.

The receipt is the payoff: each surviving claim shows a chip you can click to open the page snapshot,
highlighted to the supporting line. Not a URL you have to trust. The line.

## Abstain over guess, and the honest ceiling

The cardinal sin here is the same one veriscrape guards against: a confident, wrong answer that a
reader would trust. So citeproof is built to abstain. It cites fewer claims than a tool that will
attach any plausible-looking source, and that is the correct trade. A research agent whose every
citation is verifiable, even if it has fewer of them, is worth more than one that cites freely and is
wrong some unknown fraction of the time.

I want to be precise about what is and is not being claimed, because a tool about honesty has to be
honest about itself. Local entailment is not perfect. The published research ceiling on hard
entailment is roughly 75% balanced accuracy, and no amount of framing changes that. citeproof makes
no guarantee that every cited statement is true. It claims something narrower and testable: **every
cited claim passed a local entailment gate against a verified-OK source and is anchored to a verbatim
span, and the precision and abstention numbers are published** (see [RESULTS.md](RESULTS.md)). The
0.90 precision target is bought by aggressive abstention, not by a 90%-accurate verifier, and the
numbers say so.

## Gates before code

That honesty has to be load-bearing, not a slogan, so the project was staged around a single hard
gate. Before any user interface was built, the binder had to clear **M0**: on an evaluation set run
through the real veriscrape, citation precision at least 0.90 at an acceptable recall, with false-OK
(a junk page wrongly cited) under 2%. The thresholds were frozen on a development fold before the
held-out test fold was scored, so the result could not be tuned after the fact. If the binder could
only hit precision at near-zero recall, that would be "provable but empty," and the design would be
reconsidered rather than shipped.

M0 passed: precision 0.98 (0.96 on the hardest near-miss cases) at recall ~0.57, with false-OK 0.23%.
Only then were the fetch/verify/snapshot spine and the local web app built. The full numbers,
methodology, and honest caveats are in [RESULTS.md](RESULTS.md). Building the gate before the product
is a slower way to work, and it is the only way I know to make "provable" mean something.

---

*Written by Sanjay Chauhan, who builds reliability and data-integrity primitives for data pipelines.
citeproof is the open-source middle of a small lineage: veriscrape verifies the fetch, citeproof
verifies the claim. Apache-2.0. Reach me at san64777@gmail.com.*
