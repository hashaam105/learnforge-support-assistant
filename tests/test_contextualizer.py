"""Multi-turn context: redaction at the boundary and durable slot memory."""

from __future__ import annotations

import pytest

from app.contextualizer import Contextualizer, extract_slots, redact
from app.providers.offline import OfflineLLM


# --- redaction -------------------------------------------------------------


def test_card_number_is_removed_and_last_four_kept():
    out, kinds = redact("My card is 4111 1111 1111 1111, please refund me")
    assert "4111 1111 1111 1111" not in out
    assert "1111" in out  # last four retained for identification
    assert "card_number" in kinds


def test_security_code_is_removed():
    out, kinds = redact("the CVV is 123 if you need it")
    assert "123" not in out
    assert "security_code" in kinds


def test_email_is_reduced_to_its_domain():
    out, kinds = redact("write to me at jane.doe@university.edu")
    assert "jane.doe" not in out
    assert "university.edu" in out
    assert "email" in kinds


def test_ordinary_numbers_survive():
    """An order amount or a day count must not be mistaken for a card."""
    out, kinds = redact("I paid $79 and it has been 21 days")
    assert "$79" in out and "21 days" in out
    assert kinds == []


def test_redaction_runs_before_anything_is_stored(cfg, store):
    """The raw card number must never reach the database."""
    ctx = Contextualizer(OfflineLLM(cfg), cfg).build(
        "my card 4111111111111111 was charged", [], {}
    )
    assert "4111111111111111" not in ctx.raw_query
    assert "4111111111111111" not in ctx.standalone_query


# --- slots -----------------------------------------------------------------


def test_course_name_in_quotes_is_captured():
    assert extract_slots('I bought "Biology Essentials" last week')["course_name"] == "Biology Essentials"


def test_course_named_before_the_word_course_is_captured():
    assert extract_slots("the Physics Masterclass course vanished")["course_name"] == "Physics Masterclass"


def test_vague_course_reference_is_captured_as_a_topic():
    """TICKET-07: 'It's the biology one'."""
    assert extract_slots("It's the biology one")["course_topic"] == "biology"


def test_amount_and_platform_are_captured():
    slots = extract_slots("I was charged $79 on my iPhone")
    assert slots["amount"] == "$79"
    assert slots["platform"] == "ios"


def test_screenshot_offer_is_remembered():
    """TICKET-03 turns on whether the learner has evidence."""
    assert extract_slots("I have a screenshot of the offer")["has_screenshot"] is True


# --- turn building ---------------------------------------------------------


def test_first_turn_is_not_rewritten(cfg):
    ctx = Contextualizer(OfflineLLM(cfg), cfg).build("How do I reset my password?", [], {})
    assert ctx.standalone_query == "How do I reset my password?"
    assert not ctx.is_followup


def test_slots_persist_into_a_later_turn_without_an_llm(cfg):
    """Keyless mode cannot resolve coreference, but it must not forget entities."""
    history = [
        {"role": "user", "content": "I want a refund"},
        {"role": "assistant", "content": "Which course?"},
    ]
    ctx = Contextualizer(OfflineLLM(cfg), cfg).build(
        "the payment from last week", history, {"course_topic": "biology"}
    )
    assert "biology" in ctx.standalone_query
    assert ctx.is_followup


def test_short_follow_up_is_marked(cfg):
    history = [{"role": "user", "content": "Cancel my LearnForge"},
               {"role": "assistant", "content": "Subscription or enrollment?"}]
    ctx = Contextualizer(OfflineLLM(cfg), cfg).build("The payment", history, {})
    assert ctx.is_followup


def test_request_for_a_human_is_detected(cfg):
    ctx = Contextualizer(OfflineLLM(cfg), cfg).build(
        "just let me talk to a real person", [], {}
    )
    assert ctx.wants_human


def test_normal_question_is_not_a_human_request(cfg):
    ctx = Contextualizer(OfflineLLM(cfg), cfg).build(
        "Does a person review refund requests?", [], {}
    )
    assert not ctx.wants_human


# --- subject reference is matched by shape, not by a fixed vocabulary ------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("It's the biology one", "biology"),
        ("the astronomy one", "astronomy"),
        ("I mean the quantum-computing one", "quantum-computing"),
        ("the machine learning course", "machine learning"),
    ],
)
def test_any_subject_resolves_not_just_the_sample_ones(message, expected):
    """The subject list used to be (biology|physics|chemistry|python|ux|design|
    data) — the exact courses in the sample tickets. Any other course failed."""
    assert extract_slots(message).get("course_topic") == expected


@pytest.mark.parametrize("message", ["the last one", "the very last one", "the other one",
                                     "the first one", "the same one"])
def test_filler_words_are_not_mistaken_for_a_subject(message):
    assert "course_topic" not in extract_slots(message)
