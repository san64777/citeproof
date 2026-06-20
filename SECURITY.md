# Security Policy

## Reporting a vulnerability

citeproof fetches URLs and processes untrusted page content, and it runs a local model over that
content, so a parsing, fetch, or classification bug can affect a user's machine or the integrity of a
report. If you find a security issue, please email **san64777@gmail.com** instead of opening a public
issue. I will acknowledge within a few days and work with you on a fix and a disclosure timeline.

## What is in scope

citeproof is a self-hosted, local agent. Realistic concerns:

- A crafted page that makes the fetch, the reader, or the binder crash, hang, or consume excessive
  memory or CPU.
- A way to make the fetch layer reach an unintended target (for example, a server-side request to a
  private or internal address) through a supplied or discovered URL.
- A way to make the binder attach a citation to a claim the source does not support, or to slip a
  non-`OK` page past the veriscrape verdict gate so it becomes a citable source. A confident, wrong
  citation that a reader would trust is exactly the failure citeproof exists to prevent.

## What is not a security issue

The binder being wrong on some claim, or abstaining on a claim you think it should cite, is a
**verification report**, not a vulnerability. Abstaining when the evidence is not conclusive is by
design. Please open a normal issue with a non-sensitive reproduction (a public URL or a small
claim/source pair).

## Supported versions

citeproof is pre-release. Fixes land on `main` until there is a released version.
