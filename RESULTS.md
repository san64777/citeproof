# Results

citeproof's whole pitch is "provable," so the numbers matter more than the prose. This is the honest
record of what has been measured, how, and where the limits are. If a number here is wrong or
overstated, that is a bug; open an issue.

## The gate (M0)

The product is only as honest as its weakest binder pass, so before any UI was built, the binder had
to clear a pre-registered bar on an evaluation set:

> citation precision >= 0.90 at acceptable recall, **and** false-OK (a junk page wrongly admitted as
> citable) < 2%.

Thresholds (`tau_mc = 0.5`, `tau_db = 0.3`) were frozen on a development fold **before** the held-out
test fold was scored once.

### Citation precision and recall

| metric | result |
|---|---|
| pooled citation precision (lower bound) | **0.98** |
| precision on the hardest near-miss cases (lower bound) | **0.96** (point estimate 1.00; an adversarial audit of all 57 cited near-miss pairs found 0 over-attributions) |
| recall | **0.57** |
| held-out test fold (tune-on-dev, score-test-once) | pooled **0.97** / near-miss **0.915** / recall **0.514** -> GO |

Precision is bought with abstention: citeproof cites fewer claims so the ones it cites are right. A
high precision at near-zero recall would be "provable but empty" and was an explicit fail condition;
recall of ~0.51-0.57 clears it.

### Safety (false-OK)

The dangerous failure is citing a junk page as if it were real. Against **433** junk pages (login
walls, paywalls, soft-404s, parked domains, spun filler) run through the real veriscrape gate:

| metric | result |
|---|---|
| false-OK rate | **0.23%** (1 of 433) |
| Clopper-Pearson 95% upper bound | **1.09%** (< the 2% bar) |
| real articles that stay citable | 43 / 43 |

### End-to-end

The full production binder (MiniCheck primary + DeBERTa second signal, frozen thresholds, qwen3:8b
writer) over a 10-question benchmark, on an RTX 3060, fully offline:

- **9 / 9** answerable questions produced at least one citation (24 citations total)
- **0** false citations
- every cited claim resolved to a **working highlighted receipt**
- every blocked/paywalled page was **excluded with its verdict**
- unanswerable questions cited **nothing** (correct abstention)
- peak **8.8 GB** VRAM of 12 GB - no swap-thrash, fully offline

## Methodology and honest caveats

- **Labels are AI-generated, not hand-labeled by humans.** Each claim/source pair was labeled by
  large language models. Reliability was checked, not assumed: labels agree across **three model
  families** (Fleiss kappa **0.963**) and across two independent blind annotations (Cohen kappa
  **0.966**). This is the accepted methodology for this project; it is stated plainly here so you can
  weigh it.
- **This is citeproof's own evaluation set, not an external benchmark.** It was constructed to stress
  the binder (clean-entailed, near-miss, and adversarial buckets), generated and then independently
  verified. It is not a third-party leaderboard.
- **Local entailment has a ceiling.** Fine-tuned NLI is not a truth oracle; published research puts
  the accuracy ceiling on hard cases around 75%. The claim is therefore narrow and testable - "every
  cited claim passed a local entailment gate against a verified-OK source, anchored to a verbatim
  span" - not "cannot be wrong."
- **Coverage is bounded by what verifies.** JavaScript-only pages, paywalls, and bot-blocked sites are
  excluded by design, so encyclopedic and static sources dominate the cited set. That is the
  intended trade: a smaller, verified source set over a larger, unverifiable one.

## Reproducing

The verification gate (veriscrape) and the binder run on local models, so the pipeline is
reproducible on your own claim/source pairs. The internal evaluation harness, labeling apparatus, and
pre-registration are kept out of this repository (they are project-internal), but the binder, the
thresholds, and the production wiring are all here in `src/citeproof/`; point them at your own pairs
to measure precision/recall on data you trust.
