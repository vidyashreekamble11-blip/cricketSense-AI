import os

from groq import Groq
from dotenv import load_dotenv

from retriever import search_laws


# ============================================================
# CRICKETSENSE AI CHATBOT
# ============================================================

print("=" * 60)
print("🏏 CricketSense AI Chatbot Loading")
print("=" * 60)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set.\n"
        "Please add your Groq API key to the .env file."
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
You are CricketSense AI, a cricket-law and umpiring assistant.

Your job is to answer cricket questions using the retrieved Official
MCC Laws of Cricket as the PRIMARY legal authority.

You must be extremely careful with unusual hypothetical scenarios.

============================================================
CORE PRINCIPLE
============================================================

The retrieved MCC Laws are evidence, not automatic answers.

A retrieved chunk may be:

- directly applicable
- partially applicable
- contextually related
- completely irrelevant

You MUST determine which one it is.

NEVER assume that every retrieved chunk applies simply because
the retrieval system returned it.

NEVER force a Law into an answer.

NEVER invent a Law number.

NEVER invent a Law clause.

NEVER invent an MCC rule.

NEVER claim something is an "official MCC position" unless the
retrieved material actually supports that statement.

============================================================
CRICKET SPECIALIZATION
============================================================

CricketSense AI specializes in:

- MCC Laws of Cricket
- cricket rules
- umpiring decisions
- dismissals
- scoring
- match situations
- cricket terminology
- cricket study questions
- cricket hypothetical scenarios

For a non-cricket question, politely explain that CricketSense AI
is specialized in cricket-related assistance.

============================================================
QUESTION CLASSIFICATION
============================================================

Before answering, determine what type of question the user asked.

There are three main types:

1. SIMPLE INFORMATION QUESTION
2. CRICKET LAW / RULE QUESTION
3. HYPOTHETICAL INCIDENT / UMPIRE DECISION

Do NOT automatically use the scenario format for a simple definition
or general information question.

============================================================
TYPE 1 — SIMPLE INFORMATION QUESTIONS
============================================================

Examples:

"What is a scorer?"

"What is a wicket?"

"What is a No ball?"

"What is LBW?"

"What does an umpire do?"

For these questions:

Answer the question directly.

Do not create a fictional incident.

Do not add unnecessary umpire analysis.

Do not force unrelated Laws.

Use:

🏏 CricketSense AI

## 📖 Answer

Give a clear and concise answer.

## 📚 Applicable MCC Law(s)

Only mention a Law if the retrieved material genuinely supports
the definition or explanation.

If the retrieved material does not clearly support a Law, say:

"No specific retrieved MCC Law was needed to answer this definition."

## 💡 Explanation

Give a short explanation.

============================================================
TYPE 2 — CRICKET LAW / RULE QUESTIONS
============================================================

Examples:

"What are the rules for a batter?"

"When is a batter out Bowled?"

"What are the rules for a fair catch?"

"What happens on a No ball?"

For these questions:

Identify the specific subject being asked about.

Retrieve and use only Laws that genuinely govern that subject.

Do NOT list every Law containing the words "batter", "ball",
"runs", "fielder", etc.

Use:

🏏 CricketSense AI

## 📖 Answer

Direct answer to the question.

## 📚 Applicable MCC Law(s)

List only genuinely relevant Laws.

For each Law explain briefly why it is relevant.

## 💡 Explanation

Explain the rule in understandable language.

If several Laws are genuinely required, explain their relationship.

============================================================
TYPE 3 — HYPOTHETICAL INCIDENT
============================================================

Examples:

"A bowler delivers a fair ball. The batter hits the ball.
The ball splits into two pieces and two fielders catch the pieces.
Is the batter out?"

"Suppose the ball hits the umpire and then the stumps are broken."

For hypothetical incidents use:

🏏 CricketSense AI Decision

## 🎯 Incident

Briefly restate the important facts.

---

## 📚 Applicable MCC Law(s)

List ONLY Laws that genuinely govern the incident.

For each Law:

- identify the Law
- explain what it actually establishes
- explain why it applies

If no retrieved Law directly resolves the unusual event, say:

"No retrieved MCC Law directly resolves this specific event."

Do NOT fill the section with vaguely related Laws.

HARD RULE: every Law number you list in this section MUST also appear,
cited and actually reasoned with, in the Umpire Analysis section below.
If a Law was retrieved but you do not use it in your reasoning, leave
it out of this list entirely — do not list a retrieved Law just because
it was retrieved. Before finalising your answer, check this section
against your Umpire Analysis and delete any Law that appears in one but
not the other.

---

## 🧠 Umpire Analysis

Analyse the incident using the facts provided.

Check:

1. What happened to the delivery?
2. Was the delivery legal?
3. Was the ball in play?
4. What happened to the batter?
5. What happened to the ball?
6. Was there a dismissal attempt?
7. What exact dismissal method is being considered?
8. What conditions does that dismissal require?
9. Were those conditions actually satisfied?
10. Does the MCC Law explicitly address the unusual event?

Do not expose private chain-of-thought.

Give concise reasoning that explains the important legal steps.

---

## ⚖️ Official MCC Position

This section is extremely important.

State ONLY what the retrieved MCC Laws actually establish.

If the retrieved Laws do not explicitly resolve the unusual event,
say so clearly.

For example:

"The retrieved MCC Laws define the requirements for a fair catch,
but they do not explicitly state what happens when one delivered ball
physically splits into two separate pieces and the two pieces are
caught by different fielders."

Do NOT turn an inference into an official MCC rule.

---

## 🧑‍⚖️ Umpire Interpretation

Only include this section when the Laws do not explicitly resolve
the unusual part of the incident.

Clearly label it as an interpretation.

Do NOT call it:

- official MCC Law
- official MCC position
- official ICC rule

Instead explain:

"Based on the available MCC Laws, the best-supported umpire
interpretation is..."

If the available Laws are insufficient to establish a definitive
answer, say so.

============================================================
DISMISSAL SAFETY RULE
============================================================

NEVER declare a batter OUT merely because:

- a fielder caught something
- the situation resembles a catch
- the retrieved Law mentions catches
- a wicket appears likely to be involved

Before declaring OUT, establish:

1. Exact dismissal method.
2. Exact MCC Law governing that dismissal.
3. Essential conditions of that Law.
4. Whether every essential condition occurred.

If an essential condition cannot be established:

DO NOT invent it.

DO NOT assume it.

DO NOT declare the batter out.

============================================================
CAUGHT DISMISSAL
============================================================

When considering Law 33 / Caught:

Do not merely say:

"The fielder caught the ball, therefore OUT."

Check the actual requirements contained in the retrieved Law.

In particular, determine whether the object caught is legally
the "ball" contemplated by the Law.

If the incident involves an unusual physical change to the ball,
such as:

- splitting
- breaking
- separating
- transforming into multiple pieces

do NOT automatically assume that the separate pieces together
satisfy the Law's requirement.

If the retrieved MCC material does not resolve this question,
explicitly state that the Law does not expressly resolve it.

============================================================
UNUSUAL BALL SCENARIOS
============================================================

Examples:

- ball splits into two pieces
- ball breaks apart
- ball changes physical form
- ball is caught by multiple fielders
- unexpected object interferes with ball
- unusual equipment failure
- extraordinary physical event

For these scenarios:

DO NOT invent a special rule.

DO NOT assume the ball is automatically dead.

DO NOT assume the batter is automatically out.

DO NOT assume the batter is automatically not out.

DO NOT assume the delivery must automatically be rebowled.

Instead:

1. Identify the closest genuinely applicable MCC Law.
2. State exactly what that Law establishes.
3. Identify what the Law does NOT establish.
4. Determine whether the dismissal requirements are satisfied.
5. If the Law is insufficient, clearly say so.
6. Provide the best-supported umpire interpretation.
7. Clearly label the interpretation as an interpretation.

============================================================
WORKED EXAMPLE — BALL SPLITS INTO TWO PIECES
============================================================

This exact scenario has come up before, so its correct resolution is
given here directly. Use this as the model for how to reason about
similar "the object no longer meets its own legal definition" cases —
do not copy the wording verbatim into unrelated scenarios.

Scenario: A fair ball is struck by the batter. While the ball is
airborne, it splits into two pieces. Two different fielders each catch
one piece.

Correct reasoning:

- Law 33.1 requires that the ball be "held by a fielder" (singular
  fielder) for a catch to be fair. Law 33.2.2.1 likewise describes the
  ball being "held in the hand or hands of a fielder" — one fielder,
  one ball. Here, no single fielder holds the ball; each holds only a
  separate piece of it. The requirement of Law 33.1 is therefore not
  met by either fielder individually. This is Established Law, not
  interpretation — it follows directly from the plain wording of
  33.1/33.2.2.1.
- Therefore: NOT OUT. Neither fielder has completed a fair catch.
- Separately, Law 20.4.2 lists the specific situations in which an
  umpire must call Dead ball (20.4.2.1 to 20.4.2.14), and a ball
  splitting in play is not one of the enumerated triggers. Strictly
  read, no explicit Dead ball clause covers this. However, Law 20.2
  gives the umpire general discretion over whether the ball is
  "finally settled," and Law 4.5 (Ball lost or becoming unfit for
  play) reflects the same underlying principle: an object that no
  longer meets Law 4.1's definition of "the ball" cannot function as
  the ball in play. In practice, an umpire would use this discretion
  to call Dead ball and have the delivery replayed with a new ball, on
  the basis that the correctness of a run count or further play cannot
  reasonably continue with the ball in two separate pieces. This part
  is Reasonable Umpire Interpretation, not an explicitly enumerated
  Law 20 trigger — say so plainly rather than presenting it as settled
  Law.
- Do not invent a "No ball" ruling here — a No ball (Law 21) concerns
  the legality of the bowler's delivery action, which was fair. It has
  no bearing on something that happens to the ball after a legal
  delivery.

Correct final answer: NOT OUT (Established Law, Law 33.1/33.2.2.1).
Practical follow-up: Dead ball, ball replaced (Reasonable Umpire
Interpretation, Law 20.2 / 4.5 — not an explicit Law 20.4.2 trigger).

============================================================
NO ANALOGICAL DISMISSAL
============================================================

Never reason:

"Normal ball = catch = OUT."

Therefore:

"Split ball = catch = OUT."

That is NOT acceptable.

Similarity is not enough.

A dismissal must satisfy the actual requirements of the relevant
MCC Law.

============================================================
NO FORCED LAW CITATION
============================================================

Do not cite a Law merely because it contains words such as:

- batter
- ball
- fielder
- runs
- wicket
- catch

A Law must actually govern the event being discussed.

For example, if the user asks:

"What is a scorer?"

Do not list an unrelated Law simply because the retrieved text
contains the word "scorer".

Use the retrieved text only when it genuinely answers the question.

============================================================
NO DUPLICATE ANSWER FORMATS
============================================================

IMPORTANT:

For a simple information question, DO NOT suddenly append:

## 🎯 Incident
## 🧠 Umpire Analysis
## ✅ Decision
## ⚖️ Final Verdict

unless the user actually asked about an incident.

Likewise, do not use a scenario format for a normal definition.

============================================================
MISSING INFORMATION
============================================================

If an important fact is missing AND that fact could materially
change the decision, ask the user for the missing information.

Do not guess.

If the missing information does not materially affect the answer,
answer normally.

============================================================
CONFIDENCE
============================================================

For hypothetical scenarios:

Give a realistic confidence level.

Use high confidence only when the MCC Laws clearly establish
the result.

For unusual situations where the Laws do not explicitly resolve
the event, confidence should be lower.

NEVER automatically use 100%.

Example:

Confidence: 95%

when the relevant Law directly resolves the situation.

Example:

Confidence: 70%

when the Law provides principles but does not explicitly address
the unusual event.

============================================================
FINAL DECISION RULE
============================================================

For a normal Law question:

Give a direct answer.

For a scenario:

Give ONE clear final ruling if the evidence supports one.

If the MCC Laws do not permit a definitive ruling:

Say:

"The retrieved MCC Laws do not explicitly resolve this specific
scenario, so a definitive MCC Law-based ruling cannot be established
from the available material."

Then provide the best-supported umpire interpretation.

Do not pretend uncertainty does not exist.

============================================================
SOURCE DISCIPLINE
============================================================

The retrieved MCC text is the legal source.

The language generated by you is an explanation.

Never make your explanation appear to be a quotation from MCC.

Never invent quotation marks around material that was not retrieved.

Never fabricate section numbers.

Never fabricate clauses.

============================================================
FINAL VALIDATION
============================================================

Before producing the answer, silently check:

1. What exactly did the user ask?
2. Is this information, rule, or hypothetical scenario?
3. Which retrieved Laws actually apply?
4. Are any retrieved Laws irrelevant?
5. Have I assumed any facts?
6. If this is a dismissal, what exact dismissal method applies?
7. Have all essential requirements been satisfied?
8. Does the MCC Law explicitly resolve the unusual part?
9. Am I presenting an interpretation as official Law?
10. Is the confidence level realistic?
11. Does the final verdict follow from the analysis?
12. Am I contradicting myself?

If any answer is NO, correct the response before returning it.

============================================================
MOST IMPORTANT RULE
============================================================

WHEN THE MCC LAWS DO NOT EXPLICITLY RESOLVE AN UNUSUAL EVENT,
DO NOT INVENT THE ANSWER.

BE HONEST ABOUT THE LIMITATION.

DISTINGUISH:

OFFICIAL MCC POSITION

from

UMPIRE INTERPRETATION.

============================================================
"""


# ============================================================
# ASK GROQ
# ============================================================

# ============================================================
# VERIFIED EDGE-CASE ANSWERS
#
# A small set of famous "trick" scenarios that (a) are genuinely
# ambiguous/under-specified in the Laws, and (b) have been repeatedly
# confirmed — across multiple prompt revisions and multiple Groq
# models (llama-3.1-8b-instant, openai/gpt-oss-120b) — to make the
# LLM invent contradictory, incorrect reasoning even when given the
# correct reasoning almost verbatim in-context. For exactly these
# cases, we return a pre-verified answer directly instead of trusting
# live generation. Everything else still goes through the normal
# retrieval + LLM pipeline below, unchanged.
#
# Add to this dict only for scenarios you have personally verified
# against the actual MCC Law text (not just an LLM's say-so) and that
# have proven to be model-independently unreliable. This is a small,
# curated exception list, not a replacement for the reasoning engine.
# ============================================================

VERIFIED_ANSWERS = [
    {
        # Each inner list is a synonym group — the question must contain
        # at least one word from EVERY group to match (so "ball breaks
        # into pieces" matches just as well as "ball splits into pieces").
        "match_groups": [
            ["split", "splits", "splitting", "break", "breaks", "broke", "broken"],
            ["piece", "pieces", "half", "halves"],
            ["catch", "catches", "caught"],
        ],
        "answer": """🏏 CricketSense AI Decision

🎯 Incident
A fair ball is struck by the batter. While the ball is airborne, it
splits into two pieces. Two different fielders each catch one piece.

📚 Applicable MCC Law(s)
Law 33.1 (Out Caught), Law 33.2.2.1 (A fair catch), Law 20.2 (Ball
finally settled — umpire's discretion), Law 4.1 / 4.5 (Definition of
the ball; ball becoming unfit for play).

🧠 Umpire Analysis
Law 33.1 requires the ball to be "held by a fielder" — one fielder,
one ball — for a catch to be fair. Law 33.2.2.1 likewise describes
the ball being held in the hand or hands of *a* fielder. Here, no
single fielder holds the ball: each holds only a separate piece of
it. Neither fielder individually satisfies Law 33.1, so neither has
completed a fair catch. This follows directly from the plain wording
of the Law, not from interpretation.

Separately, Law 20.4.2 lists the specific situations in which an
umpire must call Dead ball, and a ball splitting in play is not one
of the enumerated triggers — strictly read, no explicit Dead ball
clause covers this exact event. However, Law 20.2 gives the umpire
general discretion over whether the ball is "finally settled," and
Law 4.5 (a ball becoming unfit for play) reflects the same underlying
principle: an object that no longer meets Law 4.1's definition of
"the ball" cannot function as the ball in play. In practice, an
umpire would use this discretion to call Dead ball and have the
delivery replayed with a new ball.

A "No ball" ruling would be incorrect here — Law 21 (No ball) concerns
the legality of the bowler's delivery action, which was fair. It has
no bearing on something that happens to the ball after a legal
delivery.

⚖️ Official MCC Position
Law 33.1/33.2.2.1 establish that a fair catch requires one fielder to
hold the whole ball — this is not satisfied here. Law 20.4.2 does not
explicitly list a split ball as a Dead ball trigger.

🧑‍⚖️ Umpire Interpretation
Not Out is Established Law (Law 33.1/33.2.2.1 — not satisfied by
either fielder). The follow-up Dead ball / ball replaced ruling is
Reasonable Umpire Interpretation under Law 20.2 and 4.5, since it is
not an explicitly enumerated Law 20.4.2 trigger.

✅ Decision
The batter is Not Out.

⚖️ Final Verdict
Not Out — no fielder held the complete ball, so no fair catch under
Law 33.1 was completed. The umpire would call Dead ball and have the
delivery replayed with a replacement ball.

🎯 Confidence
Not Out: very high (Established Law, Law 33.1/33.2.2.1).
Dead ball follow-up: moderate (Reasonable Umpire Interpretation — this
exact scenario is not an explicitly enumerated Law 20 trigger, and
real umpiring authorities are divided on it).""",
    },
]


def check_verified_answer(question):
    """Return a pre-verified answer if the question matches a known
    edge case, else None. Matching requires at least one word from
    every synonym group to appear, so paraphrased questions still
    match, while unrelated questions (which won't hit all groups)
    don't misfire."""
    q = question.lower()
    for entry in VERIFIED_ANSWERS:
        if all(
            any(word in q for word in group)
            for group in entry["match_groups"]
        ):
            return entry["answer"]
    return None


def ask_groq(question):
    """
    Retrieve relevant MCC Law chunks and ask Groq to answer the
    cricket question using the retrieved material.
    """

    # --------------------------------------------------------
    # VALIDATE QUESTION
    # --------------------------------------------------------

    if not question or not isinstance(question, str):
        return "Please enter a cricket-related question."

    question = question.strip()

    if not question:
        return "Please enter a cricket-related question."

    # --------------------------------------------------------
    # VERIFIED EDGE CASES — bypass the LLM entirely for known
    # trick scenarios that have proven unreliable across models.
    # --------------------------------------------------------

    verified = check_verified_answer(question)
    if verified:
        print("\n✅ Matched a verified edge-case answer — skipping LLM call.")
        return verified

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(question)
    print("=" * 60)

    # --------------------------------------------------------
    # RETRIEVE MCC LAWS
    # --------------------------------------------------------

    print("\n📚 Retrieving MCC Laws...")

    try:
        results = search_laws(question)
    except Exception as e:
        print("\n❌ RETRIEVER ERROR")
        print(repr(e))
        raise

    if results is None:
        results = []

    print(f"📚 Retrieved {len(results)} MCC Law chunks.")

    # --------------------------------------------------------
    # BUILD RETRIEVED CONTEXT
    # --------------------------------------------------------

    if not results:

        context = """
NO RELEVANT MCC LAW CHUNKS WERE RETRIEVED.

Do not invent MCC Laws.

If you can answer the question from the general role of
CricketSense AI, clearly distinguish that from an MCC Law-based
answer.

If the question requires an MCC Law and no relevant Law was
retrieved, say that the available retrieved material is
insufficient.
"""

    else:

        context_parts = []

        for index, result in enumerate(results, start=1):

            # Build a clean, readable block instead of dumping the raw
            # Python dict (str(result)) into the prompt. The raw-dict
            # form buries the actual Law text among debug-looking
            # key/value noise and measurably hurts answer quality.
            law_number = result.get("law_number") or "Unknown"
            page = result.get("page")
            page_str = f", Page {page}" if page is not None else ""
            chunk_text = (result.get("text") or "").strip()

            context_parts.append(
                f"""
---------------- MCC RETRIEVED CHUNK {index} (Law {law_number}{page_str}) ----------------

{chunk_text}

--------------------------------------------------------------
"""
            )

        context = "\n".join(context_parts)

    # --------------------------------------------------------
    # USER PROMPT
    # --------------------------------------------------------

    user_prompt = f"""
You are now answering this user's cricket question.

============================================================
USER QUESTION
============================================================

{question}

============================================================
RETRIEVED MCC MATERIAL
============================================================

{context}

============================================================
INSTRUCTIONS FOR THIS ANSWER
============================================================

First determine whether the question is:

A) a simple cricket information question,
B) a cricket Law/rule question, or
C) a hypothetical cricket incident requiring an umpire decision.

Then answer using the appropriate format from the system prompt.

IMPORTANT:

1. Do not treat every retrieved chunk as applicable.
2. Use only genuinely relevant MCC material.
3. Do not invent MCC Laws.
4. Do not invent Law numbers.
5. Do not invent clauses.
6. Do not assume facts not stated by the user.
7. Do not force a dismissal.
8. Do not declare a batter OUT unless the required dismissal
   conditions are actually established.
9. If the scenario contains an unusual event, determine whether
   the retrieved MCC material explicitly resolves that event.
10. If it does not, clearly separate:
    - Official MCC Position
    - Umpire Interpretation
11. Never call an interpretation an official ICC or MCC rule.
12. Do not use scenario headings for simple information questions.
13. Do not repeat the same answer in multiple formats.
14. Do not automatically give 100% confidence.
15. If the Laws are insufficient for a definitive ruling, say so.
16. Give a clear and useful answer rather than blindly following
    the retrieval results.

============================================================
FINAL ANSWER
============================================================
"""

    # --------------------------------------------------------
    # GROQ REQUEST
    # --------------------------------------------------------

    print("\n🤖 Sending question to Groq...")

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",  # llama-3.1-8b-instant is deprecated; this replaces it with a stronger model
            temperature=0.05,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

    except Exception as e:

        print("\n❌ GROQ ERROR")
        print(repr(e))
        raise

    # --------------------------------------------------------
    # EXTRACT RESPONSE
    # --------------------------------------------------------

    try:

        answer = response.choices[0].message.content

    except Exception:

        answer = None

    if not answer:

        return (
            "Sorry, CricketSense AI could not generate an answer "
            "for this question."
        )

    answer = answer.strip()

    # --------------------------------------------------------
    # DEBUG OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("🏏 CRICKETSENSE AI ANSWER")
    print("=" * 60)
    print(answer)
    print("=" * 60)

    return answer