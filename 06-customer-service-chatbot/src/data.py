"""The intent corpus and canned responses for the customer-service chatbot.

Why a hand-written corpus
-------------------------
The project brief calls for a structured intents dataset with greeting, working
hours, pricing, refund, order status, contact support, product information and
goodbye. There is no public, redistributable *customer-service* intent corpus of
usable size, and scraping a real support site's logs would be neither legal nor
reproducible. So the corpus is written here, in one reviewable place, with the
intent taxonomy and the phrasing variety made explicit.

Two properties matter more than raw size:

1. **Phrasing variety.** Each intent carries several paraphrases ("refund",
   "money back", "get my money back", "want a refund") because a bag-of-words
   model has to generalise across surface forms, not memorise one string.
2. **An explicit out-of-scope bucket.** The ``unrecognized`` intent is trained on
   genuine chit-chat ("what is the weather", "tell me a joke"). Without it, the
   classifier confidently maps every off-topic message to whichever support intent
   happens to share the most words, which is exactly the failure the confidence
   threshold is supposed to catch.

Company details below are fictional. The bot must never imply it can act on a
real account, take payment, or issue a refund.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from shared.errors import InvalidInputError
from shared.paths import project_dir

logger = logging.getLogger(__name__)

PROJECT_SLUG = "06-customer-service-chatbot"

#: Fictional support details, so nothing in the responses implies a real company.
COMPANY_NAME = "Northwind Electronics"
SUPPORT_HOURS = "Monday to Friday, 09:00-18:00 UTC"
SUPPORT_EMAIL = "support@northwind-electronics.example"
SUPPORT_PHONE = "+1-555-0142"
RESPONSE_TIME = "within one business day"
REFUND_WINDOW_DAYS = 30

#: Canonical intent names in display order.
INTENT_NAMES: tuple[str, ...] = (
    "greeting",
    "working_hours",
    "pricing",
    "refund",
    "order_status",
    "contact_support",
    "product_info",
    "goodbye",
    "unrecognized",
)

#: The intent used when the model's confidence is too low.
FALLBACK_INTENT = "unrecognized"

#: Human-readable intent names for the interface.
INTENT_LABELS: dict[str, str] = {
    "greeting": "Greeting",
    "working_hours": "Working hours",
    "pricing": "Pricing",
    "refund": "Refunds",
    "order_status": "Order status",
    "contact_support": "Contact support",
    "product_info": "Product information",
    "goodbye": "Goodbye",
    "unrecognized": "Not understood",
}


@dataclass(frozen=True)
class Intent:
    """One support intent with its training utterances and reply templates."""

    name: str
    label: str
    utterances: tuple[str, ...]
    responses: tuple[str, ...]
    description: str = ""

    def sample_response(self) -> str:
        """Return the first reply template."""
        return self.responses[0]


#: The full corpus. Kept in source rather than a data file so it is reviewable
#: in a diff and cannot drift from the responses the bot gives.
INTENTS: tuple[Intent, ...] = (
    Intent(
        name="greeting",
        label="Greeting",
        description="The customer opens the conversation.",
        utterances=(
            "hello",
            "hi there",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "hello there",
            "hi",
            "hey there",
            "greetings",
            "good day",
            "is anyone there",
            "hello, how are you",
            "hi, I need some help",
        ),
        responses=(
            f"Hello! You are chatting with the {COMPANY_NAME} support assistant. "
            "How can I help you today?",
            f"Hi there, welcome to {COMPANY_NAME} support. What can I do for you?",
        ),
    ),
    Intent(
        name="working_hours",
        label="Working hours",
        description="The customer asks when support is available.",
        utterances=(
            "what are your opening hours",
            "when are you open",
            "what time do you close",
            "are you open on weekends",
            "what are your business hours",
            "when is support available",
            "do you work on saturdays",
            "what hours are you available",
            "opening times",
            "when can I reach you",
            "are you open today",
            "what are your support hours",
        ),
        responses=(
            f"Our support team is available {SUPPORT_HOURS}. Outside those hours you can "
            f"leave a message and we will reply {RESPONSE_TIME}.",
            f"We are open {SUPPORT_HOURS}. Messages sent outside those hours are answered "
            f"{RESPONSE_TIME}.",
        ),
    ),
    Intent(
        name="pricing",
        label="Pricing",
        description="The customer asks about prices, plans or discounts.",
        utterances=(
            "how much does it cost",
            "what is the price",
            "how much is a subscription",
            "do you offer a discount",
            "what does the pro plan cost",
            "pricing information",
            "are there any discounts available",
            "how much for the premium plan",
            "what plans do you have",
            "is there a free trial",
            "tell me about your pricing",
            "what does it cost per month",
        ),
        responses=(
            "Our plans start at $9.99 per month for the Starter tier. Pro is $24.99 per "
            "month and Teams is $49.99 per month. Annual billing saves 20%, and every paid "
            "plan includes a 14-day free trial.",
            "Starter is $9.99/month, Pro is $24.99/month and Teams is $49.99/month, billed "
            "monthly or annually with a 20% discount. All paid plans have a 14-day trial.",
        ),
    ),
    Intent(
        name="refund",
        label="Refunds",
        description="The customer wants money back.",
        utterances=(
            "I want a refund",
            "how do I get my money back",
            "can I have a refund please",
            "I would like my money back",
            "please refund my payment",
            "how do refunds work",
            "what is your refund policy",
            "can I return an item and get money back",
            "I was charged twice, please refund",
            "cancel my subscription and refund me",
            "I want my money back for this order",
            "refund please",
        ),
        responses=(
            f"Refunds are available within {REFUND_WINDOW_DAYS} days of purchase. This bot "
            "cannot process a refund itself - a support agent will need to action it. "
            f"Please share your order reference with us at {SUPPORT_EMAIL}.",
            f"I can help you with the policy but not issue the refund myself. Within "
            f"{REFUND_WINDOW_DAYS} days, contact {SUPPORT_EMAIL} with your order reference "
            "and an agent will process it.",
        ),
    ),
    Intent(
        name="order_status",
        label="Order status",
        description="The customer asks where their order is.",
        utterances=(
            "where is my order",
            "track my package",
            "has my order shipped",
            "what is the status of my delivery",
            "when will my order arrive",
            "order tracking",
            "my parcel has not arrived",
            "can you check my delivery",
            "how long does shipping take",
            "i want to track my shipment",
            "is my order still processing",
        ),
        responses=(
            "Orders ship within 1-2 business days and standard delivery takes 3-5 business "
            f"days. To see live status, use the tracking link in your shipping confirmation "
            f"email, or contact {SUPPORT_EMAIL} with your order number.",
            "Standard delivery takes 3-5 business days after dispatch. The tracking link in "
            f"your confirmation email shows live updates; an agent can also help at "
            f"{SUPPORT_EMAIL}.",
        ),
    ),
    Intent(
        name="contact_support",
        label="Contact support",
        description="The customer wants a way to reach a human.",
        utterances=(
            "how do I contact support",
            "can I speak to a human",
            "I need to talk to a real person",
            "what is your phone number",
            "give me an email address",
            "let me talk to an agent",
            "how can I reach customer service",
            "is there a live chat option",
            "connect me to a person",
            "who do I call for help",
        ),
        responses=(
            f"You can reach us at {SUPPORT_EMAIL} or {SUPPORT_PHONE}. A live agent is "
            f"available {SUPPORT_HOURS}.",
            f"Email {SUPPORT_EMAIL} or call {SUPPORT_PHONE}. Live agents are on duty "
            f"{SUPPORT_HOURS}.",
        ),
    ),
    Intent(
        name="product_info",
        label="Product information",
        description="The customer asks about features, compatibility or warranty.",
        utterances=(
            "what does the product do",
            "tell me about the features",
            "is it compatible with my device",
            "how much storage do I get",
            "what is included in the plan",
            "does it work offline",
            "what is the warranty",
            "can I use it on multiple devices",
            "what are the technical specifications",
            "is there a mobile app",
            "tell me more about the product",
            "does it work on my phone",
            "can I use it on my mobile",
            "will this run on android",
            "is there an app for my phone",
        ),
        responses=(
            f"{COMPANY_NAME} devices include offline mode, encrypted sync and mobile apps "
            "for iOS and Android. All hardware carries a 24-month warranty, and paid plans "
            "include 100 GB of storage with unlimited devices.",
            "Every plan includes offline mode, encrypted sync and iOS/Android apps, plus "
            "100 GB of storage and unlimited devices. Hardware has a 24-month warranty.",
        ),
    ),
    Intent(
        name="goodbye",
        label="Goodbye",
        description="The customer ends the conversation.",
        utterances=(
            "bye",
            "goodbye",
            "see you",
            "thanks, bye",
            "thank you, goodbye",
            "that is all I needed",
            "have a good day",
            "catch you later",
            "ok bye",
            "nice one, thanks",
            "that will be all",
            "that is everything, thanks",
            "thanks, that is all",
            "nothing else thanks",
        ),
        responses=(
            f"Thanks for chatting with {COMPANY_NAME}. Have a great day!",
            "Glad I could help. Take care!",
        ),
    ),
    Intent(
        name="unrecognized",
        label="Not understood",
        description=(
            "Off-topic chit-chat. Training on this keeps the confidence score honest "
            "instead of mapping every odd message onto a support intent."
        ),
        utterances=(
            "what is the weather like",
            "tell me a joke",
            "who won the football match",
            "play some music",
            "what time is it",
            "can you sing a song",
            "what is two plus two",
            "tell me about the stock market",
            "translate this to french",
            "what is your favourite colour",
            "give me a recipe",
            "who are you",
            "what does the fox say",
            "help me with my homework",
        ),
        responses=(
            "I am not confident I understood that. Please contact support or rephrase your "
            "question.",
            "Sorry, that is outside what I can help with. Please rephrase your question or "
            f"contact {SUPPORT_EMAIL}.",
        ),
    ),
)

INTENTS_BY_NAME: dict[str, Intent] = {intent.name: intent for intent in INTENTS}

#: Class order used everywhere, so the label list never depends on dict ordering.
CLASS_LABELS: list[str] = list(INTENT_NAMES)


def get_intent(name: str) -> Intent:
    """Return an :class:`Intent` by name.

    Raises:
        InvalidInputError: When the name is not a known intent.
    """
    intent = INTENTS_BY_NAME.get(name)
    if intent is None:
        raise InvalidInputError(
            f"Unknown intent '{name}'.",
            hint=f"Choose one of: {', '.join(INTENT_NAMES)}.",
        )
    return intent


def response_for(intent_name: str, *, index: int = 0) -> str:
    """Return one reply template for an intent.

    Args:
        intent_name: Canonical intent name.
        index: Which template to return; wraps around if out of range.
    """
    templates = get_intent(intent_name).responses
    return templates[int(index) % len(templates)]


def build_corpus() -> "Any":
    """Return the training corpus as ``(texts, labels)``.

    Every utterance is augmented with a small set of neutral prefixes and
    suffixes so the classifier sees phrasing variation rather than memorising
    exact strings. The augmentation is deterministic - it is a fixed list, not
    random - so training stays reproducible.
    """
    prefixes = ("", "hi, ", "hello, ", "hey, ", "question: ", "quick question - ")
    suffixes = ("", " please", " thanks", "?", "?")

    texts: list[str] = []
    labels: list[str] = []
    for intent in INTENTS:
        for utterance in intent.utterances:
            for prefix in prefixes:
                for suffix in suffixes:
                    texts.append(f"{prefix}{utterance}{suffix}")
                    labels.append(intent.name)
    return texts, labels


def corpus_summary() -> dict[str, Any]:
    """Return corpus statistics for the dataset panel."""
    texts, labels = build_corpus()
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return {
        "n_intents": len(INTENTS),
        "n_base_utterances": sum(len(intent.utterances) for intent in INTENTS),
        "n_training_examples": len(texts),
        "examples_per_intent": counts,
        "base_per_intent": {intent.name: len(intent.utterances) for intent in INTENTS},
        "company": COMPANY_NAME,
        "support_hours": SUPPORT_HOURS,
        "support_email": SUPPORT_EMAIL,
    }


#: Held-out paraphrases used to check the confidence gate.
#:
#: These are written in a different register from the training utterances
#: ("a real person", "entry level option", "send money yesterday"), so the gate
#: is measured on phrasing it has not memorised. Each pair carries its true
#: intent, which is what makes the report meaningful.
GATE_PROBES: tuple[tuple[str, str], ...] = (
    ("I would like to speak to an actual person please", "contact_support"),
    ("what are the hours when your team is available", "working_hours"),
    ("could you tell me the cost of the entry level option", "pricing"),
    ("I sent money yesterday and would like it returned", "refund"),
    ("any idea where my parcel is right now", "order_status"),
    ("do you have an application for my phone", "product_info"),
    ("thanks very much, catch you", "goodbye"),
    ("morning there", "greeting"),
    ("what is the capital of Portugal", FALLBACK_INTENT),
    ("please sing me a lullaby", FALLBACK_INTENT),
    ("can you give me a recipe for bread", FALLBACK_INTENT),
)


def suggested_messages() -> list[str]:
    """Return ready-to-send example messages for the interface."""
    return [
        "Hi there!",
        "What are your opening hours?",
        "How much does the Pro plan cost?",
        "I would like a refund please",
        "Where is my order?",
        "Can I speak to a human?",
        "Is there a mobile app?",
        "Thanks, goodbye",
        "What is the weather like today?",
    ]


#: Prompts shown in the interface's data source location.
def readme_location() -> Any:
    """Return the path of the project README, used for the error hints."""
    return project_dir(PROJECT_SLUG) / "README.md"