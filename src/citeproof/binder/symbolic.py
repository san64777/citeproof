"""The ORTHOGONAL signal: a pure-Python CONTRADICTION detector (not an entailment checker).

This is the genuinely-independent second check that ANDs with MiniCheck (the pre-registered
second-signal gate). RoBERTa and Flan-T5 share the MiniCheck recipe, so a second same-family NLI buys
little on near-miss; near-miss failures usually turn on a flipped NUMBER, a swapped YEAR, a
dropped "not", a flipped QUANTIFIER, a flipped direction word, or a swapped ENTITY. A symbolic
check catches exactly those and an NLI of the same lineage does not.

Crucial contract: this returns ok=False ONLY when it finds a CONCRETE contradiction. Absence of
a contradiction passes (ok=True). It does NOT try to prove entailment - that is MiniCheck's job.
Distinguishing "unsupported" from "contradicted" matters: only the latter may fire here, because
firing on mere absence of evidence would steal entailment's role and crater recall. The detector
is deliberately CONSERVATIVE: when unsure, it prefers ok=True to avoid false flags. A missed
contradiction is backstopped by MiniCheck; a false flag silently drops a good citation, so every
rule is biased toward precision-of-the-contradiction (few false flags).

Implementation: regex plus small hand-built lexicons. No spaCy, no heavy NLP dependency; this
module imports only the standard library, so the binder core needs no new runtime dep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SymbolicResult:
    """Outcome of the contradiction check.

    ok is False ONLY when a concrete contradiction is found; contradictions lists a short,
    human-readable reason per contradiction (empty when ok is True).
    """

    ok: bool
    contradictions: list[str] = field(default_factory=list)


# --- NUMBERS -----------------------------------------------------------------

# A number token: integer or decimal, with optional thousands separators stripped later.
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# A percentage: "12%" or "12 %" or "12 percent" / "12 per cent".
_PERCENT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:%|percent|per\s*cent)", re.IGNORECASE)

# Unit words we treat as making a bare number "salient" enough to compare. Kept small and
# concrete; a number with a unit in BOTH claim and span is a strong, checkable signal.
_UNIT_WORDS = {
    "tonne",
    "tonnes",
    "ton",
    "tons",
    "kg",
    "kilograms",
    "kilogram",
    "g",
    "grams",
    "gram",
    "metre",
    "metres",
    "meter",
    "meters",
    "km",
    "kilometre",
    "kilometres",
    "mile",
    "miles",
    "dollar",
    "dollars",
    "euro",
    "euros",
    "pound",
    "pounds",
    "people",
    "survivors",
    "patients",
    "units",
    "votes",
    "seats",
    "points",
    "degrees",
    "years",
    "months",
    "days",
    "hours",
}

# Approximation cues: when one of these precedes a number in the CLAIM, an exact match in the
# span is not required; a span number within a tolerance band is accepted.
_APPROX_CUES = {
    "about",
    "approximately",
    "approx",
    "around",
    "roughly",
    "nearly",
    "circa",
    "ca",
    "almost",
    "some",
    "~",
}

# Bound (inequality) cues: a number preceded by one of these is a one-sided BOUND, not an exact
# value, so it is not symbolically checkable for equality. When a bound cue immediately precedes a
# number on EITHER side, that number is skipped from the exact/tolerance comparison (abstain), since
# "more than 100" is consistent with "150" and "fewer than 50" is consistent with "30". Stored as
# token tuples (longest phrase first) so the last 1-2 tokens before the number can be matched.
_BOUND_CUES: list[tuple[str, ...]] = [
    # Lower bounds.
    ("more", "than"),
    ("greater", "than"),
    ("at", "least"),
    ("minimum", "of"),
    ("upwards", "of"),
    ("over",),
    ("above",),
    # Upper bounds.
    ("fewer", "than"),
    ("less", "than"),
    ("no", "more", "than"),
    ("at", "most"),
    ("up", "to"),
    ("maximum", "of"),
    ("under",),
    ("below",),
]


def _norm_number(token: str) -> str:
    """Normalize a numeric token for comparison ('1,200' -> '1200', '3.0' -> '3')."""
    token = token.replace(",", "")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token or "0"


def _to_float(norm: str) -> float:
    try:
        return float(norm)
    except ValueError:
        return float("nan")


def _decimals(norm: str) -> int:
    """Number of decimal places in a normalized number string."""
    return len(norm.split(".", 1)[1]) if "." in norm else 0


def _approx_tolerant_match(claim_vals: set[str], span_vals: set[str], approx: bool) -> bool:
    """True if every claim value has a matching span value (exact, or within a tolerance band
    when approx is set). Used so an approximation hedge ('about 50') accepts a near span number.
    """
    for cv in claim_vals:
        if cv in span_vals:
            continue
        if not approx:
            return False
        cf = _to_float(cv)
        if cf != cf:  # NaN
            return False
        # Tolerance: round to the claim's precision, or +-0.5 for an integer claim, whichever is
        # wider. "about 50" accepts 48..52ish; "about 12%" accepts 12.0..12.x within rounding.
        cdec = _decimals(cv)
        if cdec > 0:
            band = 0.5 * (10 ** -cdec)
        else:
            # Integer claim: a relative band of 5% (min 0.5) tolerates rounding like 50 ~ 48.
            band = max(0.5, abs(cf) * 0.05)
        matched = False
        for sv in span_vals:
            sf = _to_float(sv)
            if sf == sf and abs(sf - cf) <= band:
                matched = True
                break
        if not matched:
            return False
    return True


def _approx_before(text: str, num_start: int) -> bool:
    """True if an approximation cue immediately precedes the number at num_start."""
    prefix = text[:num_start].lower()
    if prefix.rstrip().endswith("~"):
        return True
    tokens = re.findall(r"[a-z~]+", prefix)
    return bool(tokens) and tokens[-1] in _APPROX_CUES


def _bound_before(text: str, num_start: int) -> bool:
    """True if a bound (inequality) cue immediately precedes the number at num_start.

    Mirrors _approx_before but matches multi-token phrases ("more than", "at least", "no more
    than", "up to"). The cue must be the LAST token(s) before the number, so only an adjacent
    bound triggers abstention.
    """
    prefix = text[:num_start].lower()
    tokens = re.findall(r"[a-z]+", prefix)
    if not tokens:
        return False
    for cue in _BOUND_CUES:
        n = len(cue)
        if len(tokens) >= n and tuple(tokens[-n:]) == cue:
            return True
    return False


def _percentages_approx(text: str) -> tuple[set[str], bool]:
    """Return percentage values and whether ANY of them carried an approximation cue.

    A percentage immediately preceded by a bound cue ("up to 80%", "over 60%") is a one-sided
    bound, not an exact value, so it is SKIPPED (abstain): it never enters the comparison set.
    """
    vals: set[str] = set()
    approx = False
    for m in _PERCENT_RE.finditer(text):
        if _bound_before(text, m.start(1)):
            continue
        vals.add(_norm_number(m.group(1)))
        if _approx_before(text, m.start(1)):
            approx = True
    return vals, approx


def _years(text: str) -> set[str]:
    """Four-digit YEAR tokens with a date cue, excluding numbers that are really counts.

    A four-digit number is only a year when (a) it is not a unit-number (e.g. '2000 patients'
    is a count, not a year) and (b) a date cue (in/since/by/during/around/founded/born/year, or
    a month name) sits just before it, or it is a clearly-modern year preceded by no other cue.
    Conservative: only 1000-2999.
    """
    units = _unit_numbers(text)
    out: set[str] = set()
    lowered = text.lower()
    for m in re.finditer(r"\b(1\d{3}|2\d{3})\b", text):
        norm = _norm_number(m.group(0))
        if norm in units:
            continue
        prefix = lowered[: m.start()]
        cue_tokens = re.findall(r"[a-z]+", prefix)
        last = cue_tokens[-1] if cue_tokens else ""
        if last in _DATE_CUES or last in _MONTHS:
            out.add(m.group(0))
    return out


_DATE_CUES = {
    "in",
    "since",
    "by",
    "during",
    "around",
    "before",
    "after",
    "founded",
    "born",
    "year",
    "established",
    "from",
    "until",
    "circa",
}


def _unit_numbers(text: str) -> set[str]:
    """Numbers that are immediately followed by a known unit word (number-with-units)."""
    out: set[str] = set()
    lowered = text.lower()
    for m in _NUM_RE.finditer(lowered):
        tail = lowered[m.end() : m.end() + 24]
        word_match = re.match(r"\s+([a-z]+)", tail)
        if word_match and word_match.group(1) in _UNIT_WORDS:
            out.add(_norm_number(m.group(0)))
    return out


def _unit_numbers_with_approx(text: str) -> tuple[set[str], bool]:
    """Unit-numbers plus whether ANY of them carried an approximation cue.

    A unit-number immediately preceded by a bound cue ("more than 100 people", "fewer than 50
    patients") is a one-sided bound, not an exact value, so it is SKIPPED (abstain): it never
    enters the comparison set.
    """
    out: set[str] = set()
    approx = False
    lowered = text.lower()
    for m in _NUM_RE.finditer(lowered):
        tail = lowered[m.end() : m.end() + 24]
        word_match = re.match(r"\s+([a-z]+)", tail)
        if word_match and word_match.group(1) in _UNIT_WORDS:
            if _bound_before(text, m.start()):
                continue
            out.add(_norm_number(m.group(0)))
            if _approx_before(text, m.start()):
                approx = True
    return out, approx


# --- NEGATION / POLARITY -----------------------------------------------------

# Single-token negation cues. We compare the negation status near a SHARED content word: if the
# claim asserts X and the span asserts "not X" (or vice versa) for the same anchor, that is a flip.
_NEG_CUES = {
    "not",
    "no",
    "never",
    "none",
    "nor",
    "cannot",
    "didn't",
    "doesn't",
    "don't",
    "isn't",
    "wasn't",
    "weren't",
    "aren't",
    "won't",
    "couldn't",
    "wouldn't",
    "shouldn't",
    "hasn't",
    "haven't",
    "hadn't",
    "without",
    "unable",
}

# Multi-token negation cues ("fails to", "unable to", "failed to"). Detected as a phrase so a
# bare "fails"/"unable" elsewhere does not over-fire.
_NEG_PHRASES = [
    ("fails", "to"),
    ("fail", "to"),
    ("failed", "to"),
    ("failing", "to"),
    ("unable", "to"),
]

# Stopwords that never make a useful shared content anchor for the negation rule.
_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "they",
    "them",
    "their",
    "we",
    "he",
    "she",
    "his",
    "her",
    "this",
    "that",
    "these",
    "those",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "by",
    "with",
    "and",
    "but",
    "or",
    "as",
    "from",
    "into",
    "can",
    "could",
    "will",
    "would",
    "may",
    "might",
    "must",
    "should",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
    "not",
    "no",
    "never",
    "fails",
    "fail",
    "failed",
    "failing",
    "unable",
    "without",
    "than",
    "then",
    "there",
    "here",
    "which",
    "who",
    "what",
    "when",
    "where",
    "all",
    "any",
    "some",
    "over",
    "under",
    "about",
    "said",
}

_WORD_RE = re.compile(r"[a-z']+")

# Word tokens plus the clause-break punctuation ',' and ';' kept as their own tokens, so the
# quantifier and negation rules can stop a scan at a clause boundary.
_CLAUSE_TOK_RE = re.compile(r"[a-z']+|[,;]")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


# Verb lemma families so "did not fall" vs "fell" line up despite differing surface forms.
_VERB_FAMILIES = [
    {"fall", "falls", "fell", "fallen", "falling"},
    {"rise", "rises", "rose", "risen", "rising"},
    {"grow", "grows", "grew", "grown", "growing"},
    {"increase", "increases", "increased", "increasing"},
    {"decline", "declines", "declined", "declining"},
    {"drop", "drops", "dropped", "dropping"},
    {"win", "wins", "won", "winning"},
    {"pass", "passes", "passed", "passing"},
    {"work", "works", "worked", "working"},
    {"reduce", "reduces", "reduced", "reducing"},
    {"lower", "lowers", "lowered", "lowering"},
    {"detect", "detects", "detected", "detecting"},
]


# Map every verb-family member to a single canonical family key, so irregular forms such as
# 'fell' and 'fall' share an anchor that a suffix stemmer alone would miss.
_FAMILY_KEY: dict[str, str] = {}
for _family in _VERB_FAMILIES:
    _rep = sorted(_family, key=len)[0]
    for _member in _family:
        _FAMILY_KEY[_member] = f"#{_rep}"


def _stem(word: str) -> str:
    """A crude suffix stemmer so 'reduces'/'reducing'/'reduced' share a key with 'reduce'.

    Verb-family members (including irregulars like 'fell'->'fall') map to a shared family key.
    """
    if word in _FAMILY_KEY:
        return _FAMILY_KEY[word]
    for suf in ("ing", "ed", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            base = word[: -len(suf)]
            # 'reduc' (from reducing/reduced) -> normalize a trailing consonant cluster lightly.
            return base
    return word


def _proposition_words(text: str, exclude: frozenset[str]) -> set[str]:
    """Stemmed content words of text (plus digit numbers), dropping stopwords, short tokens, and the
    rule's own OPERATOR words in `exclude` (negation cues, or quantifiers).

    Used to require that the claim and span describe the SAME proposition before a lexical
    contradiction rule fires. A shared verb / head / quantifier with a DIFFERENT object or predicate
    is NOT a contradiction ("treat bacterial" vs "not treat viral"; "all employees got bonuses" vs
    "no employees were laid off"; the "not ... until" idiom), and these blind-anchor false flags are
    the cardinal sin (a false flag drops a good citation). Digit numbers are content too, so a
    differing count/year distinguishes two propositions.
    """
    words = {_stem(w) for w in _words(text) if w not in _STOPWORDS and w not in exclude and len(w) >= 3}
    nums = {_norm_number(m.group(0)) for m in _NUM_RE.finditer(text)}
    return words | nums


def _same_proposition(claim: str, span: str, exclude: frozenset[str]) -> bool:
    """True iff every content word/number in the claim (minus the operator words) also appears in the
    span - i.e. the claim adds nothing the span lacks, so they are about the same proposition.
    """
    return not (_proposition_words(claim, exclude) - _proposition_words(span, exclude))


# Clause-break tokens: a negation cue does not reach an anchor across one of these. Punctuation
# ',' and ';' are kept as tokens by _CLAUSE_TOK_RE; the conjunctions split coordinate clauses.
_CLAUSE_BREAK_TOKENS = {",", ";", "and", "but", "or", "while", "although", "though", "whereas"}

# Phrase-negation heads ("fails to", "unable to"): the first member of each _NEG_PHRASES pair.
_NEG_PHRASE_HEADS = {a for a, _ in _NEG_PHRASES}


def _content_anchor_occurrences(words: list[str]) -> dict[str, list[int]]:
    """Map a content (non-stopword) stem to the indices of ALL its occurrences.

    All occurrences (not just the first) so the negation rule can require EVERY span occurrence of an
    anchor to disagree with the claim before firing. Clause-break punctuation tokens are skipped.
    """
    out: dict[str, list[int]] = {}
    for i, w in enumerate(words):
        if w in _CLAUSE_BREAK_TOKENS or w in _STOPWORDS or len(w) < 3:
            continue
        key = _stem(w)
        if len(key) < 3:
            continue
        out.setdefault(key, []).append(i)
    return out


def _negated_adjacent(words: list[str], idx: int) -> bool:
    """True if a negation cue DIRECTLY governs the anchor at idx (within 1 token to its left).

    Strict adjacency (no clause break possible across a single adjacent token): either the token
    immediately to the left is a single-token negation cue ("not fall", "no benefit"), or it is the
    'to' of a 'fails to'/'unable to' phrase whose head sits one token further left ("fails to
    reduce"). Anything looser (a cue separated by another word, or across a clause boundary) does NOT
    count as directly governing - we ABSTAIN there.
    """
    if idx - 1 >= 0 and words[idx - 1] in _NEG_CUES:
        return True
    # Phrase cue "fails to <anchor>": token to the left is 'to', the one before it is a phrase head.
    if idx - 2 >= 0 and words[idx - 1] == "to" and words[idx - 2] in _NEG_PHRASE_HEADS:
        return True
    return False


def _has_negation_cue(words: list[str]) -> bool:
    """True if the token list carries ANY negation cue (single-token or a 'fails to' phrase)."""
    for i, w in enumerate(words):
        if w in _NEG_CUES:
            return True
        if w == "to" and i - 1 >= 0 and words[i - 1] in _NEG_PHRASE_HEADS:
            return True
    return False


def _negation_flip(claim: str, span: str) -> list[str]:
    """A polarity contradiction anchored on a SHARED non-stopword content lemma.

    CONSERVATIVE: fire only when ONE side has a negation cue DIRECTLY governing the shared anchor
    (adjacent, within 1 token to its left, no clause break) AND the OTHER side has NO negation cue
    anywhere in its text (it is confidently positive). If BOTH sides contain any negation cue, or the
    negation is not directly adjacent on the firing side, ABSTAIN. This kills scope-over-modifier
    false flags ("no significant difference" vs "no difference"; "did not really help" vs "did not
    help" - both negated, so abstain) while keeping "sales did not fall" vs "sales fell" firing.
    """
    out: list[str] = []
    cwords = _CLAUSE_TOK_RE.findall(claim.lower())
    swords = _CLAUSE_TOK_RE.findall(span.lower())
    c_has_neg = _has_negation_cue(cwords)
    s_has_neg = _has_negation_cue(swords)
    # If both sides carry any negation cue, the polarity is too entangled to judge -> abstain.
    if c_has_neg and s_has_neg:
        return out
    # Exactly one side may be the negated ("firing") side; the other must be confidently positive.
    if not c_has_neg and not s_has_neg:
        return out
    # SHARED-PROPOSITION GUARD: a shared negated verb is a polarity flip only if the REST of the
    # proposition matches. "Antibiotics treat bacterial infections" vs "...do not treat viral
    # infections" (same verb, different object), the "not ... until" idiom, and stem collisions
    # ("housing"/"houses") are NOT contradictions. Require the claim's content (minus negation
    # operators) to be present in the span before firing.
    if not _same_proposition(claim, span, frozenset(_NEG_CUES)):
        return out
    canch = _content_anchor_occurrences(cwords)
    sanch = _content_anchor_occurrences(swords)
    shared = sorted(set(canch) & set(sanch))
    for key in shared:
        if c_has_neg:
            # Claim is the firing side: negation must directly govern EVERY claim occurrence of the
            # anchor (unambiguous), and the span (positive side) must not negate it anywhere.
            if not all(_negated_adjacent(cwords, i) for i in canch[key]):
                continue
            out.append(f"polarity flip on '{key}': claim negates it, span asserts it positively")
        else:
            # Span is the firing side; claim is the confidently-positive side.
            if not all(_negated_adjacent(swords, i) for i in sanch[key]):
                continue
            out.append(f"polarity flip on '{key}': span negates it, claim asserts it positively")
    return out


# --- DIRECTION ANTONYMS ------------------------------------------------------

# Directional movement pairs: claim has one pole, span the opposite -> contradiction. Each entry
# maps a lemma to a sign (+1 up, -1 down); a claim/span pair with opposite signs on the SAME scale
# is a flip. The gate is the SUBJECT region (content tokens before the verb) and the moved OBJECT
# token (content token just after the verb): both must match, so we are sure it is the SAME quantity
# moving in opposite directions and not two different things (Revenue rose vs Unemployment fell).
_DIRECTION_SIGN: dict[str, int] = {}
for _up, _down in [
    ("rise", "fall"),
    ("rises", "falls"),
    ("rose", "fell"),
    ("risen", "fallen"),
    ("rising", "falling"),
    ("grow", "shrink"),
    ("grows", "shrinks"),
    ("grew", "shrank"),
    ("growing", "shrinking"),
    ("grow", "decline"),
    ("grew", "declined"),
    ("growing", "declining"),
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("increased", "decreased"),
    ("increasing", "decreasing"),
    ("increase", "drop"),
    ("increased", "dropped"),
    ("gain", "lose"),
    ("gains", "loses"),
    ("gained", "lost"),
    ("expand", "contract"),
    ("expands", "contracts"),
    ("expanded", "contracted"),
    ("climb", "drop"),
    ("climbs", "drops"),
    ("climbed", "dropped"),
    ("climbing", "dropping"),
    ("jump", "plunge"),
    ("jumped", "plunged"),
    ("surge", "plunge"),
    ("surged", "plunged"),
]:
    _DIRECTION_SIGN[_up] = 1
    _DIRECTION_SIGN[_down] = -1


# Manner adverbs that decorate a direction verb without being the moved object; skipped when
# locating the object so "increased sharply" and "rose steadily" expose the real object (if any).
_DIRECTION_ADVERBS = {
    "sharply",
    "steadily",
    "slightly",
    "significantly",
    "considerably",
    "markedly",
    "dramatically",
    "modestly",
    "again",
    "further",
    "overall",
}


def _direction_regions(words: list[str]) -> tuple[int | None, frozenset[str], str | None]:
    """For the FIRST direction verb in words, return (verb index, subject-region content tokens,
    moved-object token). The subject region is the content tokens before the verb; the object is the
    nearest content token after the verb (skipping stopwords/adverbs/numbers), or None.
    """
    vi = next((i for i, w in enumerate(words) if w in _DIRECTION_SIGN), None)
    if vi is None:
        return None, frozenset(), None
    pre = frozenset(
        w
        for w in words[:vi]
        if w not in _STOPWORDS and w not in _DIRECTION_SIGN and len(w) >= 2
    )
    obj: str | None = None
    for j in range(vi + 1, min(len(words), vi + 4)):
        w = words[j]
        if w in _STOPWORDS or w in _DIRECTION_SIGN or w in _DIRECTION_ADVERBS or len(w) < 2:
            continue
        obj = _stem(w)
        break
    return vi, pre, obj


def _direction_antonym(claim: str, span: str) -> list[str]:
    """Fire ONLY on a high-confidence same-quantity direction flip.

    The claim and span must each carry a direction verb of OPPOSITE sign, AND the subject region
    (content tokens before the verb) must be EQUAL and non-empty on both sides, AND the moved object
    (content token just after the verb) must match (both absent, or both equal). This excludes years
    and index numbers (they are not word tokens) and refuses to fire when the two texts describe
    different quantities: Revenue rose vs Unemployment fell, exports vs imports, hiring vs footprint.
    """
    cwords = _words(claim)
    swords = _words(span)
    _cvi, cpre, cobj = _direction_regions(cwords)
    _svi, spre, sobj = _direction_regions(swords)
    if not cpre or cpre != spre:
        return []
    c_signs = {_DIRECTION_SIGN[w] for w in cwords if w in _DIRECTION_SIGN}
    s_signs = {_DIRECTION_SIGN[w] for w in swords if w in _DIRECTION_SIGN}
    # Opposite poles only: no overlap, and each side carries a real direction sign.
    if c_signs & s_signs or not c_signs or not s_signs:
        return []
    # Moved-object gate: both absent or both equal. If exactly one side names an object, the things
    # moving differ ("increased hiring" vs "decreased footprint") -> not the same quantity.
    if (cobj is None) != (sobj is None):
        return []
    if cobj is not None and cobj != sobj:
        return []
    return [f"direction flip: claim signs {sorted(c_signs)} vs span signs {sorted(s_signs)}"]


# --- QUANTIFIERS -------------------------------------------------------------

# Quantifier scale tiers (0 = none, 1 = partial, 2 = universal). A claim at a tier disjoint from
# the span on the SAME scale is a contradiction (e.g. "all" vs "some"). Conservative: requires
# both a claim quantifier and a span quantifier.
_QUANT_TIER = {
    "all": 2,
    "every": 2,
    "always": 2,
    "everyone": 2,
    "everything": 2,
    "entirely": 2,
    "completely": 2,
    "most": 1,
    "majority": 1,
    "some": 1,
    "several": 1,
    "many": 1,
    "few": 1,
    "minority": 1,
    "sometimes": 1,
    "occasionally": 1,
    "partly": 1,
    "partially": 1,
    "none": 0,
    "no": 0,
    "never": 0,
    "nobody": 0,
    "nothing": 0,
    "neither": 0,
}

# Direct quantifier-antonym pairs (case-insensitive), independent of the tier table. These fire
# even when both members live in the same coarse tier (most vs few), because they are genuine
# opposites of degree.
_QUANT_ANTONYMS = [
    ("most", "few"),
    ("most", "minority"),
    ("majority", "minority"),
    ("many", "few"),
    ("all", "none"),
    ("all", "no"),
    ("every", "none"),
    ("everyone", "nobody"),
    ("always", "never"),
]

# Contrast conjunctions: a competing quantifier introduced after one of these is a SEPARATE clause
# about a different predicate ("most supported ... while few opposed"), not a flip of the claim.
_CONTRAST_CONJ = {
    "but",
    "although",
    "though",
    "while",
    "whereas",
    "however",
    "yet",
}

# Tokens we skip when looking for the head noun a quantifier modifies (determiners, of, copulas,
# auxiliaries). The first content token past these is the quantifier's head.
_QUANT_SKIP = {
    "of",
    "the",
    "a",
    "an",
    "its",
    "their",
    "his",
    "her",
    "our",
    "them",
    "it",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "did",
    "do",
    "does",
    "been",
    "being",
    "be",
    "that",
    "this",
    "these",
    "those",
    "to",
}


def _quant_head(words: list[str], qi: int, window: int = 4) -> str | None:
    """The head noun a quantifier at index qi modifies: the nearest following content token
    (skipping determiners/copulas/of) within a small window. Stems it so 'patients' anchors stably.
    """
    for j in range(qi + 1, min(len(words), qi + 1 + window)):
        w = words[j]
        if w in _QUANT_SKIP or w in _QUANT_TIER or len(w) < 2:
            continue
        return _stem(w)
    return None


def _quant_competes(a: str, b: str) -> bool:
    """True if quantifiers a and b are genuine opposites: a direct antonym pair, or disjoint tiers."""
    if a == b:
        return False
    for x, y in _QUANT_ANTONYMS:
        if (a == x and b == y) or (a == y and b == x):
            return True
    return _QUANT_TIER[a] != _QUANT_TIER[b]


def _quantifier_flip(claim: str, span: str) -> list[str]:
    """Fire ONLY on a high-confidence same-head quantifier flip.

    The claim's quantifier and the span's competing quantifier must modify the SAME (stemmed) head
    noun, the competing span quantifier must NOT sit in a contrast clause (after but/although/while/
    ',' /';'), and if the span repeats the claim's own quantifier+head verbatim the head is confirmed
    and we NEVER fire on it. This kills false flags like "most supported ... while few opposed" and
    "all flights resumed with no delays" while keeping "all studies / some studies" firing.
    """
    # SHARED-PROPOSITION GUARD: a shared quantified head noun is a flip only if the PREDICATE matches.
    # "All employees received bonuses" vs "No employees were laid off" (same head, different predicate)
    # is not a contradiction. Require the claim's content (minus quantifier words) to be in the span.
    if not _same_proposition(claim, span, frozenset(_QUANT_TIER)):
        return []
    cw = _words(claim)

    # Claim quantifier -> head-noun pairs.
    cpairs: list[tuple[str, str]] = []
    for i, w in enumerate(cw):
        if w in _QUANT_TIER:
            h = _quant_head(cw, i)
            if h is not None:
                cpairs.append((w, h))
    if not cpairs:
        return []

    # Span quantifier -> (head, index, in-contrast-clause) triples. Tokenize keeping the clause-break
    # punctuation ',' and ';' as their own tokens so a contrast clause can be detected.
    span_toks = _CLAUSE_TOK_RE.findall(span.lower())
    spairs: list[tuple[str, str, bool]] = []
    contrast_seen = False
    for idx, tok in enumerate(span_toks):
        if tok in {",", ";"} or tok in _CONTRAST_CONJ:
            contrast_seen = True
            continue
        if tok in _QUANT_TIER:
            h = _quant_head(span_toks, idx)
            if h is not None:
                spairs.append((tok, h, contrast_seen))
    if not spairs:
        return []

    for cq, ch in cpairs:
        # Span repeats the claim's quantifier+head verbatim -> head confirmed, never fire on it.
        if any(sq == cq and sh == ch for sq, sh, _ in spairs):
            continue
        for sq, sh, in_contrast in spairs:
            if sh != ch or in_contrast:
                continue
            if _quant_competes(cq, sq):
                return [f"quantifier flip on '{ch}': claim '{cq}' vs span '{sq}'"]
    return []


# --- BARE DATES / ORDINALS (cue-gated) ---------------------------------------

_MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}

# Index nouns: "<noun> <number>" pins a numbered reference. Mismatched numbers under the SAME noun
# is an off-by-one contradiction.
_INDEX_NOUNS = {
    "chapter",
    "section",
    "figure",
    "fig",
    "table",
    "page",
    "step",
    "article",
    "clause",
    "appendix",
    "volume",
    "part",
}

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
# "<month> <day>" e.g. "June 7" and "<day> <month>" e.g. "7 March".
_MONTH_DAY_RE = re.compile(rf"\b({_MONTH_ALT})\s+(\d{{1,2}})\b", re.IGNORECASE)
_DAY_MONTH_RE = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_ALT})\b", re.IGNORECASE)
_INDEX_ALT = "|".join(sorted(_INDEX_NOUNS, key=len, reverse=True))
_INDEX_RE = re.compile(rf"\b({_INDEX_ALT})\s+(\d{{1,4}})\b", re.IGNORECASE)


def _month_day_map(text: str) -> dict[str, set[str]]:
    """Map a month name -> set of day numbers seen next to it (either order)."""
    out: dict[str, set[str]] = {}
    for m in _MONTH_DAY_RE.finditer(text):
        month = m.group(1).lower()
        day = m.group(2)
        if 1 <= int(day) <= 31:
            out.setdefault(month, set()).add(str(int(day)))
    for m in _DAY_MONTH_RE.finditer(text):
        day = m.group(1)
        month = m.group(2).lower()
        if 1 <= int(day) <= 31:
            out.setdefault(month, set()).add(str(int(day)))
    return out


def _index_map(text: str) -> dict[str, set[str]]:
    """Map an index noun -> set of numbers seen after it."""
    out: dict[str, set[str]] = {}
    for m in _INDEX_RE.finditer(text):
        out.setdefault(m.group(1).lower(), set()).add(str(int(m.group(2))))
    return out


def _date_ordinal_contradiction(claim: str, span: str) -> list[str]:
    out: list[str] = []
    cmd = _month_day_map(claim)
    smd = _month_day_map(span)
    for month in set(cmd) & set(smd):
        if cmd[month].isdisjoint(smd[month]):
            out.append(
                f"date mismatch on {month}: claim {sorted(cmd[month])} vs span {sorted(smd[month])}"
            )
    cix = _index_map(claim)
    six = _index_map(span)
    for noun in set(cix) & set(six):
        if cix[noun].isdisjoint(six[noun]):
            out.append(
                f"index mismatch on {noun}: claim {sorted(cix[noun])} vs span {sorted(six[noun])}"
            )
    return out


# --- NUMBERS: comparison -----------------------------------------------------


def _number_contradiction(claim: str, span: str) -> list[str]:
    out: list[str] = []

    cp, cp_approx = _percentages_approx(claim)
    sp, sp_approx = _percentages_approx(span)
    # An approximation hedge on EITHER side (claim "about 12%" OR span "about 98 survivors") widens
    # the tolerance: a hedged value should not contradict a nearby exact one regardless of which side
    # carried the hedge.
    if cp and sp and not _approx_tolerant_match(cp, sp, cp_approx or sp_approx):
        out.append(f"percentage mismatch: claim {sorted(cp)} vs span {sorted(sp)}")

    cy = _years(claim)
    sy = _years(span)
    # Only fire if a claim year is genuinely ABSENT from the span. A span that LISTS several years
    # (the claim's plus others, even where the claim's is not cue-gated as a year there) is consistent
    # - the other years are different facts, not a contradiction ("supernova in 1054 ... pulsar in
    # 1968" supports a claim about 1054).
    span_nums = {_norm_number(m.group(0)) for m in _NUM_RE.finditer(span)}
    if cy and sy and cy.isdisjoint(sy) and not (cy & span_nums):
        out.append(f"year mismatch: claim {sorted(cy)} vs span {sorted(sy)}")

    cu, cu_approx = _unit_numbers_with_approx(claim)
    su, su_approx = _unit_numbers_with_approx(span)
    if cu and su and not _approx_tolerant_match(cu, su, cu_approx or su_approx):
        out.append(f"unit-number mismatch: claim {sorted(cu)} vs span {sorted(su)}")

    return out


# --- NAMED ENTITIES (DISABLED) -----------------------------------------------

# DISABLED: entity-swap detection deferred to MiniCheck. A regex entity matcher cannot resolve
# aliases/abbreviations/titles (WHO=World Health Organization, President Biden=Joe Biden) without
# real NER, and false-flagging a correct paraphrase is worse than missing an entity swap (which
# MiniCheck backstops). The entity rule was the single largest false-flag source (capitalized common
# nouns, abbreviations, titles), so it is inert: the
# functions below ALWAYS return no contradictions and are kept only so the dispatcher stays stable.


def _entity_contradiction(claim: str, span: str) -> list[str]:
    """DISABLED entity-swap check: always returns no contradictions (deferred to MiniCheck)."""
    return []


def _single_entity_contradiction(claim: str, span: str) -> list[str]:
    """DISABLED single-token entity-swap check: always returns no contradictions."""
    return []


# --- WORD-FORM FRACTIONS (EVALUATED, NOT SHIPPED) ----------------------------

# A spelled-out fraction mismatch ("a quarter" vs "a fifth" of the same whole) is a real near-miss
# the numeric rule misses (no digit), and it was prototyped here (2026-06-09) after a live case slipped
# both MiniCheck and this check. It was REMOVED, not shipped, for the same reason the entity rule is
# disabled: it cannot be made reliable symbolically. To catch the real case the span must be allowed to
# REWORD around the fraction ("land" -> "total land area"); but that same flexibility false-flags the
# COMMON and consistent "partition" pattern, two fractions slicing one whole into different parts
# ("a third of the budget to salaries" vs "a quarter of the budget to marketing"). Two adversarial
# red-team rounds confirmed the false-flag class is intrinsic (a same-head guard, then a content-word +
# number guard, each left a fresh class: stopword-only distinguishers over/under, here/there, his/her,
# and stem collisions housing/houses). Distinguishing "reworded same whole" from "different slice"
# needs meaning, not regex. Per the module contract (a false flag silently drops a good citation and is
# strictly worse than a miss) and partitions being common in cited quantitative text, the rule does net
# harm, so word-form fraction mismatches are DEFERRED TO MiniCheck.


# --- TOP-LEVEL ---------------------------------------------------------------


def symbolic_consistency(claim: str, span: str) -> SymbolicResult:
    """Return ok=False ONLY on a concrete contradiction between claim and span; else ok=True.

    Checks, in order, for: a number/percentage/unit mismatch (approximation-tolerant, with
    bound-hedge abstention), a cue-gated year mismatch, a bare date/ordinal off-by-one, a
    conservative negation/polarity flip on a shared content lemma, a direction-antonym flip, and a
    quantifier flip/antonym. Entity-swap detection is DISABLED (deferred to MiniCheck): the entity
    calls below are inert. The ABSENCE of any contradiction is a pass (ok=True): this detector never
    tries to prove entailment.
    """
    contradictions: list[str] = []
    contradictions += _number_contradiction(claim, span)
    contradictions += _date_ordinal_contradiction(claim, span)
    contradictions += _negation_flip(claim, span)
    contradictions += _direction_antonym(claim, span)
    contradictions += _quantifier_flip(claim, span)
    contradictions += _entity_contradiction(claim, span)
    contradictions += _single_entity_contradiction(claim, span)
    return SymbolicResult(ok=not contradictions, contradictions=contradictions)
