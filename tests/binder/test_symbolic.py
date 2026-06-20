"""Tests for the ORTHOGONAL symbolic contradiction detector.

This detector is a CONTRADICTION finder, not a positive-entailment checker. It returns
ok=False ONLY when it finds a concrete contradiction between claim and span (a flipped
number, a swapped year, a dropped "not", a quantifier flip, a swapped salient entity).
Absence of contradiction passes (ok=True): proving entailment is MiniCheck's job, not this
detector's. The bar for ok=False is HIGH; when unsure, it prefers ok=True to avoid false flags.

Every example named in the build spec is covered here: the contradiction cases assert
ok==False, and each matching non-contradiction case asserts ok==True.
"""

from __future__ import annotations

from citeproof.binder.symbolic import SymbolicResult, symbolic_consistency


# --- NUMBERS / PERCENTAGES ---------------------------------------------------


def test_percentage_contradiction() -> None:
    r = symbolic_consistency("Revenue grew 12%", "Revenue grew 21% last year")
    assert r.ok is False
    assert r.contradictions  # a reason is recorded


def test_percentage_match_with_extra_context_is_ok() -> None:
    # Same number, span just adds context -> NOT a contradiction (entailment's job).
    r = symbolic_consistency("Revenue grew 12%", "Revenue grew 12% in 2025")
    assert r.ok is True
    assert r.contradictions == []


def test_count_contradiction() -> None:
    r = symbolic_consistency("There were 7 survivors", "There were 9 survivors")
    assert r.ok is False


def test_number_spelled_percent_word_contradiction() -> None:
    # "12 percent" spelled form vs a different percentage -> contradiction.
    r = symbolic_consistency("It rose 12 percent", "It rose 20 percent over the period")
    assert r.ok is False


def test_number_with_units_contradiction() -> None:
    r = symbolic_consistency("The bridge carries 50 tonnes", "The bridge carries 15 tonnes")
    assert r.ok is False


def test_number_with_units_match_is_ok() -> None:
    r = symbolic_consistency("The bridge carries 50 tonnes", "The bridge can carry 50 tonnes of load")
    assert r.ok is True


def test_same_percent_word_and_symbol_is_ok() -> None:
    # "12 percent" in the claim, "12%" in the span: same value, not a contradiction.
    r = symbolic_consistency("It grew 12 percent", "Sales grew 12% that quarter")
    assert r.ok is True


# --- YEARS / DATES -----------------------------------------------------------


def test_year_contradiction() -> None:
    r = symbolic_consistency("The company was founded in 1998", "The company was founded in 1989")
    assert r.ok is False


def test_year_match_is_ok() -> None:
    r = symbolic_consistency(
        "The company was founded in 1998",
        "Founded in 1998, the company grew quickly.",
    )
    assert r.ok is True


# --- NEGATION / POLARITY -----------------------------------------------------


def test_negation_contradiction_effective() -> None:
    r = symbolic_consistency("The treatment is effective", "The treatment is not effective")
    assert r.ok is False


def test_negation_contradiction_did_not_fall_vs_fell() -> None:
    r = symbolic_consistency("Sales did not fall", "Sales fell sharply in the quarter")
    assert r.ok is False


def test_polarity_match_is_ok() -> None:
    r = symbolic_consistency("The treatment is effective", "The treatment is effective in trials")
    assert r.ok is True


# --- QUANTIFIERS -------------------------------------------------------------


def test_quantifier_all_vs_some() -> None:
    r = symbolic_consistency("All studies agree", "Some studies agree on the point")
    assert r.ok is False


def test_quantifier_always_vs_sometimes() -> None:
    r = symbolic_consistency("The drug always works", "The drug sometimes works")
    assert r.ok is False


def test_quantifier_none_vs_some() -> None:
    r = symbolic_consistency("None of the patients improved", "Some of the patients improved")
    assert r.ok is False


def test_quantifier_match_is_ok() -> None:
    r = symbolic_consistency("All studies agree", "All studies agree on the finding")
    assert r.ok is True


# --- NAMED-ENTITY (conservative) ---------------------------------------------


def test_entity_swap_now_deferred_to_minicheck() -> None:
    # Entity-swap detection is DISABLED (deferred to MiniCheck): a regex entity matcher
    # cannot resolve aliases/abbreviations/titles without real NER, and false-flagging a
    # correct paraphrase is worse than missing an entity swap (which MiniCheck backstops).
    # Previously asserted ok=False; now ok=True.
    r = symbolic_consistency("Acme acquired Beta", "Acme acquired Gamma last month")
    assert r.ok is True
    assert r.contradictions == []


def test_entity_match_is_ok() -> None:
    r = symbolic_consistency("Acme acquired Beta", "Acme acquired Beta in a cash deal")
    assert r.ok is True


def test_entity_absent_but_no_competitor_is_ok() -> None:
    # Beta is absent but there is no competing capitalized entity to contradict it.
    # Be CONSERVATIVE: prefer ok=True (entailment will judge support).
    r = symbolic_consistency("Acme acquired Beta", "The acquisition closed quickly.")
    assert r.ok is True


# --- REGRESSION: ENTITY FALSE-FIRE (defect 1) --------------------------------


def test_sentence_initial_common_noun_not_entity_approximately() -> None:
    # "Approximately"/"Roughly" are sentence-initial capitalized COMMON words, not entities.
    r = symbolic_consistency("Approximately 100 attended", "Roughly 100 attended")
    assert r.ok is True
    assert r.contradictions == []


def test_sentence_initial_determiner_words_not_entities() -> None:
    r = symbolic_consistency("Yesterday the vote passed", "Today the vote passed")
    assert r.ok is True
    assert r.contradictions == []


def test_sentence_initial_role_nouns_not_entities() -> None:
    r = symbolic_consistency("Researchers found support", "Scientists found support")
    assert r.ok is True
    assert r.contradictions == []


def test_the_role_phrase_not_entity_mismatch() -> None:
    # "The Labor Department" must not be treated as a competing entity vs "Unemployment".
    r = symbolic_consistency(
        "Unemployment fell to 4% in May.",
        "The Labor Department said the jobless rate dropped to 4% in May.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_distractor_leading_words_not_entities() -> None:
    for word_c, word_s in [
        ("Revenue", "Sales"),
        ("Sales", "Revenue"),
        ("Production", "Output"),
        ("Employment", "Hiring"),
    ]:
        r = symbolic_consistency(f"{word_c} climbed steadily", f"{word_s} climbed steadily")
        assert r.ok is True, f"{word_c} vs {word_s} should not flag"
        assert r.contradictions == []


# --- REGRESSION: YEAR vs COUNT COLLISION (defect 2) --------------------------


def test_count_2000_patients_vs_year_2019_is_ok() -> None:
    # "2000 patients" is a count (unit-number), not a year; "2019" is a year. No contradiction.
    r = symbolic_consistency("The study enrolled 2000 patients", "The study ran in 2019")
    assert r.ok is True
    assert r.contradictions == []


# --- REGRESSION: QUANTIFIER most-vs-few (defect 3) ---------------------------


def test_quantifier_most_vs_few_lowercase() -> None:
    r = symbolic_consistency("most patients recovered", "few patients recovered")
    assert r.ok is False
    assert r.contradictions


def test_quantifier_majority_vs_minority() -> None:
    r = symbolic_consistency("the majority approved", "the minority approved")
    assert r.ok is False


def test_quantifier_all_vs_none_antonym() -> None:
    r = symbolic_consistency("all of them survived", "none of them survived")
    assert r.ok is False


# --- REGRESSION: NEGATION on content lemma (defect 4) ------------------------


def test_negation_fails_to_reduce_mortality() -> None:
    r = symbolic_consistency("The drug reduces mortality", "The drug fails to reduce mortality")
    assert r.ok is False


def test_negation_unable_to_detect() -> None:
    r = symbolic_consistency("The system can detect threats", "The system is unable to detect threats")
    assert r.ok is False


def test_negation_fails_to_lower() -> None:
    r = symbolic_consistency("It lowers risk", "It fails to lower risk")
    assert r.ok is False


# --- REGRESSION: ANTONYM DIRECTION FLIP (defect 5) --------------------------


def test_direction_antonym_rose_vs_fell() -> None:
    r = symbolic_consistency("Sales rose", "Sales fell")
    assert r.ok is False


def test_direction_antonym_grew_vs_declined() -> None:
    r = symbolic_consistency("Revenue grew", "Revenue declined")
    assert r.ok is False


def test_direction_antonym_increased_vs_dropped() -> None:
    r = symbolic_consistency("Prices increased", "Prices dropped")
    assert r.ok is False


def test_direction_antonym_rose_vs_fell_same_magnitude() -> None:
    r = symbolic_consistency("Sales rose 5%", "Sales fell 5%")
    assert r.ok is False


def test_direction_antonym_same_pole_is_ok() -> None:
    # Same direction word on both sides -> NOT a contradiction.
    r = symbolic_consistency("Sales rose 5%", "Sales rose 5% last quarter")
    assert r.ok is True
    assert r.contradictions == []


# --- REGRESSION: BARE DATE / ORDINAL off-by-one (defect 6) ------------------


def test_month_day_off_by_one() -> None:
    r = symbolic_consistency("held on June 7", "held on June 8")
    assert r.ok is False


def test_day_month_off_by_one() -> None:
    r = symbolic_consistency("on 7 March", "on 8 March")
    assert r.ok is False


def test_index_noun_chapter_off_by_one() -> None:
    r = symbolic_consistency("chapter 3 covers it", "chapter 4 covers it")
    assert r.ok is False


def test_month_day_match_is_ok() -> None:
    r = symbolic_consistency("held on June 7", "the event held on June 7 this year")
    assert r.ok is True
    assert r.contradictions == []


# --- REGRESSION: ROUNDING / APPROXIMATION HEDGE (defect 7) ------------------


def test_approx_percentage_within_tolerance_is_ok() -> None:
    r = symbolic_consistency("About 12% of users churned", "Exactly 12.3% of users churned")
    assert r.ok is True
    assert r.contradictions == []


def test_approx_unit_number_within_tolerance_is_ok() -> None:
    r = symbolic_consistency("The bridge carries about 50 tonnes", "The bridge carries 48 tonnes")
    assert r.ok is True
    assert r.contradictions == []


def test_approx_does_not_excuse_large_gap() -> None:
    # "about 50" should still contradict 80: outside any reasonable tolerance band.
    r = symbolic_consistency("The bridge carries about 50 tonnes", "The bridge carries 80 tonnes")
    assert r.ok is False


# --- REGRESSION: ABBREVIATION / SYNONYM entity (defect 8) -------------------


def test_uk_united_kingdom_abbrev_is_ok() -> None:
    r = symbolic_consistency("The UK left the bloc", "The United Kingdom left the bloc in 2020")
    assert r.ok is True
    assert r.contradictions == []


# --- REGRESSION: phantom multi-word entity "In Paris" (defect 10) -----------


def test_in_paris_phantom_entity_is_ok() -> None:
    # "In Paris" must reduce to "Paris"; the same city on both sides is no contradiction.
    r = symbolic_consistency("In Paris the summit opened", "In Paris the summit opened on time")
    assert r.ok is True
    assert r.contradictions == []


# --- ABSENCE OF EVIDENCE IS NOT A CONTRADICTION ------------------------------


def test_no_checkable_feature_unrelated_span_is_ok() -> None:
    # A claim with no number/year/quantifier/negation/salient-entity hook vs an unrelated
    # span: the detector must NOT fire. Distinguishing unsupported from contradicted is
    # entailment's job, not the symbolic check's.
    r = symbolic_consistency(
        "The garden was pleasant in the morning",
        "The committee reconvened to discuss the budget.",
    )
    assert r.ok is True
    assert r.contradictions == []


# --- REGRESSION: LEXICAL RULES MUST NOT FALSE-FLAG ENTAILED PAIRS ------------
# These pairs are clearly entailed / consistent. A lexical antonym/negation rule that
# fires here is a recall-killer (worse than a miss). They must NEVER be flagged.


def test_quantifier_contrast_clause_most_few_is_ok() -> None:
    # Span repeats the claim's "most ... supported" verbatim; "few opposed" is a contrast
    # clause about a DIFFERENT predicate. Not a contradiction.
    r = symbolic_consistency(
        "Most respondents supported the measure.",
        "Most respondents supported the measure, while few opposed it.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_quantifier_all_with_no_delays_is_ok() -> None:
    # "no delays" is not a quantifier flip against "all flights"; "all flights resumed"
    # is repeated verbatim in the span.
    r = symbolic_consistency(
        "All flights resumed.",
        "All flights resumed with no delays reported.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_quantifier_many_few_contrast_clause_is_ok() -> None:
    r = symbolic_consistency(
        "Many species are at risk.",
        "Many species are at risk, although few have gone extinct.",
    )
    assert r.ok is True
    assert r.contradictions == []


# --- SHARED-PROPOSITION GUARD (false-flag regression; audit 2026-06-09) ----------------------------
# A shared negated verb or quantified head with a DIFFERENT object/predicate is NOT a contradiction.


def test_negation_different_object_is_ok() -> None:
    # Same subject+verb, DIFFERENT object: both statements are simultaneously true.
    r = symbolic_consistency(
        "Antibiotics treat bacterial infections.",
        "Antibiotics do not treat viral infections.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_negation_not_until_idiom_is_ok() -> None:
    # The span CONFIRMS the claim via "not founded until 1917" (= founded in 1917). The negation rule
    # must not read this idiom as a polarity flip. Real eval pair (ce-63) that was being false-dropped.
    r = symbolic_consistency(
        "The Stanley Cup predates the founding of the National Hockey League.",
        "Because the National Hockey League was not founded until 1917, the trophy is older than the league.",
    )
    assert r.ok is True


def test_negation_stem_collision_is_ok() -> None:
    # "housing" and "houses" collide under the crude stemmer; the proposition guard keeps two unrelated
    # facts from false-flagging.
    r = symbolic_consistency("The housing market is strong.", "No houses were sold last week.")
    assert r.ok is True


def test_quantifier_different_predicate_is_ok() -> None:
    # Same head noun "employees", DIFFERENT predicate (received bonuses vs laid off): consistent.
    r = symbolic_consistency("All employees received bonuses.", "No employees were laid off.")
    assert r.ok is True
    assert r.contradictions == []


def test_year_present_in_multi_year_span_is_ok() -> None:
    # The span lists several years; the claim's year (1054) is present even though ANOTHER year (1968)
    # is cue-gated. Not a contradiction - the other year is a different fact (real eval pair ah-2).
    r = symbolic_consistency(
        "The Crab Nebula is the remnant of a supernova observed in the year 1054.",
        "The Crab Nebula coincides with that 1054 guest star; its pulsar was identified in 1968.",
    )
    assert r.ok is True


def test_approx_hedge_on_span_side_is_tolerated() -> None:
    # The hedge may sit on the SPAN side: "100 survivors" vs "about 98 survivors" is not a mismatch.
    r = symbolic_consistency("There were 100 survivors.", "There were about 98 survivors.")
    assert r.ok is True
    # but a real gap still fires regardless of the hedge.
    assert symbolic_consistency("There were 100 survivors.", "There were about 80 survivors.").ok is False


def test_direction_different_subjects_revenue_unemployment_is_ok() -> None:
    # Revenue rose vs Unemployment fell: different subjects, same year. Both can be true.
    r = symbolic_consistency("Revenue rose in 2020.", "Unemployment fell in 2020.")
    assert r.ok is True
    assert r.contradictions == []


def test_direction_shared_index_number_different_subject_is_ok() -> None:
    # "group 5" vs "district 5" share only the index number 5; subjects differ. Not a flip.
    r = symbolic_consistency("In group 5, scores rose.", "In district 5, scores fell.")
    assert r.ok is True
    assert r.contradictions == []


def test_direction_different_subjects_hiring_footprint_is_ok() -> None:
    r = symbolic_consistency(
        "The company increased hiring this year.",
        "The company decreased its carbon footprint this year.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_direction_exports_imports_same_year_is_ok() -> None:
    # Year 2020 is shared but exports != imports; the shared number is a YEAR, not a magnitude.
    r = symbolic_consistency(
        "In 2020, exports increased sharply.",
        "In 2020, imports decreased sharply.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_negation_not_surprising_confirmed_is_ok() -> None:
    # "not surprising" negates a different word; "confirmed the theory" is asserted on BOTH sides.
    r = symbolic_consistency(
        "The results were not surprising and confirmed the theory.",
        "The results confirmed the theory.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_negation_did_not_increase_costs_but_increased_revenue_is_ok() -> None:
    # The span both negates and asserts "increase"; it also asserts the claim's "increased revenue"
    # verbatim. EVERY occurrence does not disagree, so no flip.
    r = symbolic_consistency(
        "The policy increased revenue.",
        "The policy did not increase costs, but it did increase revenue.",
    )
    assert r.ok is True
    assert r.contradictions == []


# --- REGRESSION GUARDS: TRUE CONTRADICTIONS MUST STILL FIRE ------------------


def test_guard_revenue_rose_vs_fell_same_magnitude_year() -> None:
    r = symbolic_consistency(
        "Revenue rose 10 percent in 2020",
        "Revenue fell 10 percent in 2020",
    )
    assert r.ok is False


def test_guard_all_vs_none_samples() -> None:
    r = symbolic_consistency("All samples tested positive", "None of the samples tested positive")
    assert r.ok is False


def test_guard_percentage_grew_12_vs_21() -> None:
    r = symbolic_consistency("revenue grew 12%", "revenue grew 21%")
    assert r.ok is False


def test_guard_entity_swap_beta_gamma_now_deferred_to_minicheck() -> None:
    # Entity-swap detection is DISABLED: a regex entity matcher cannot resolve
    # aliases/abbreviations/titles without real NER, and a false flag on a correct
    # paraphrase is worse than a miss (MiniCheck backstops entity swaps). So this
    # asserts ok=True now; the entity contradiction is deferred to MiniCheck.
    r = symbolic_consistency("Acme acquired Beta", "Acme acquired Gamma")
    assert r.ok is True
    assert r.contradictions == []


def test_guard_negation_sales_did_not_fall_vs_fell() -> None:
    r = symbolic_consistency("sales did not fall", "sales fell")
    assert r.ok is False


def test_guard_year_1998_vs_1989() -> None:
    r = symbolic_consistency("founded in 1998", "founded in 1989")
    assert r.ok is False


# --- REGRESSION: FINAL CONVERGENCE PASS (must NEVER false-flag) --------------
# Clearly-entailed / consistent pairs the detector must NOT flag. A false flag here
# silently drops a good citation; MiniCheck backstops any real miss. These exercise the
# conservative-negation rule, the number bound-hedge abstention, and disabled entity-swap.


# Negation made conservative: both sides carry a negation cue -> ABSTAIN (no flip).
def test_negation_no_significant_difference_both_negated_is_ok() -> None:
    r = symbolic_consistency(
        "There was no significant difference between groups.",
        "The study found no difference between the two groups.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_negation_no_major_effect_vs_no_effect_is_ok() -> None:
    r = symbolic_consistency(
        "The treatment had no major effect.",
        "The treatment had no effect.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_negation_no_clear_benefit_vs_no_benefit_is_ok() -> None:
    r = symbolic_consistency(
        "Researchers found no clear benefit.",
        "Researchers found no benefit.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_negation_not_statistically_significant_both_negated_is_ok() -> None:
    r = symbolic_consistency(
        "The result was not statistically significant.",
        "The difference was not significant.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_negation_did_not_really_help_vs_did_not_help_is_ok() -> None:
    r = symbolic_consistency(
        "The policy did not really help.",
        "The policy did not help the economy.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_negation_no_observed_adverse_event_is_ok() -> None:
    r = symbolic_consistency(
        "There was no observed adverse event.",
        "There were no adverse events.",
    )
    assert r.ok is True
    assert r.contradictions == []


# Number bound-hedge abstention: a bound cue ("more than", "over", "at least", "fewer
# than", "up to") next to a number means the exact value is not symbolically checkable.
def test_number_more_than_100_vs_150_is_ok() -> None:
    r = symbolic_consistency(
        "More than 100 people attended.",
        "A total of 150 people attended the rally.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_number_at_least_1000_vs_1500_is_ok() -> None:
    r = symbolic_consistency(
        "At least 1,000 people signed.",
        "Some 1,500 people signed the petition.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_number_over_200_vs_240_is_ok() -> None:
    r = symbolic_consistency(
        "Over 200 patients were treated.",
        "In total, 240 patients were treated.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_number_fewer_than_50_vs_30_is_ok() -> None:
    r = symbolic_consistency(
        "Fewer than 50 patients relapsed.",
        "Only 30 patients relapsed.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_number_up_to_80pct_vs_60pct_is_ok() -> None:
    r = symbolic_consistency(
        "Up to 80% of cases resolve.",
        "Around 60% of cases resolve on their own.",
    )
    assert r.ok is True
    assert r.contradictions == []


# Entity-swap disabled (deferred to MiniCheck): aliases/abbreviations/titles must NOT flag.
def test_entity_who_world_health_organization_is_ok() -> None:
    r = symbolic_consistency(
        "The World Health Organization issued guidance.",
        "WHO issued new guidance on the outbreak.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_entity_ecb_european_central_bank_is_ok() -> None:
    r = symbolic_consistency(
        "The European Central Bank raised rates.",
        "The ECB raised interest rates by a quarter point.",
    )
    assert r.ok is True
    assert r.contradictions == []


def test_entity_president_biden_joe_biden_is_ok() -> None:
    r = symbolic_consistency(
        "President Biden signed the bill.",
        "Joe Biden signed the infrastructure bill.",
    )
    assert r.ok is True
    assert r.contradictions == []


# --- RESULT SHAPE ------------------------------------------------------------


def test_result_shape() -> None:
    r = symbolic_consistency("anything", "anything")
    assert isinstance(r, SymbolicResult)
    assert isinstance(r.ok, bool)
    assert isinstance(r.contradictions, list)
