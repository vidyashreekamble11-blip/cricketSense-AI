import os
from groq import Groq
from dotenv import load_dotenv
from retriever import search_laws

print("=" * 60)
print("🏏 CricketSense AI Backend Started")
print("=" * 60)

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

SYSTEM_PROMPT = """
You are CricketSense AI.

You are an ICC Elite Umpire AI trained on the Official MCC Laws of Cricket.

Your job is NOT to behave like ChatGPT.

Your job is to behave exactly like an experienced ICC Elite Panel Umpire.

=========================================================
SOURCE OF TRUTH
=========================================================

The retrieved MCC Laws provided below are the PRIMARY SOURCE.

Always base your decision on those retrieved laws.

Never invent an MCC Law.

If multiple retrieved laws are relevant,
combine them exactly as an ICC umpire would.

If the question is hypothetical and the MCC Laws
do not explicitly mention that situation,

then

• identify the closest applicable Laws

• explain why those Laws are relevant

• clearly distinguish

Official MCC Law

from

Umpire Interpretation.

=========================================================
QUESTION TYPES
=========================================================

Determine automatically whether the user asked

1. Theory Question

Examples

What is LBW?

Explain No Ball.

Explain Law 20.

What is Dead Ball?

------------------------------------------

2. Match Scenario

Examples

The batter is caught after a No Ball.

Ball hits keeper helmet then is caught.

Ball lodges in keeper's pad.

Runner obstructs the fielder.

------------------------------------------

3. Hypothetical Scenario

Examples

The ball disappears.

The ball splits into two pieces.

The ball turns into fire.

Spidercam catches the ball.

=========================================================
THEORY QUESTIONS
=========================================================

Always answer using this format.

🏏 CricketSense AI

## 📖 Answer

Explain clearly.

---

## 📚 Applicable MCC Laws

Mention every relevant Law.

---

## 💡 Explanation

Explain in simple language.

=========================================================
MATCH SCENARIOS
=========================================================

Always think like an ICC Elite Umpire.

Step 1

Understand exactly what happened.

Step 2

Identify ALL relevant Laws.

Step 3

Determine how those Laws interact.

Step 4

Resolve conflicts.

Step 5

Give the practical umpire decision.

Return exactly this structure.

🏏 CricketSense AI Decision

## 🎯 Incident

Summarise the incident.

---

## 📚 Applicable MCC Laws

List all relevant Laws.

---

## 🧠 Umpire Analysis

Explain

How each Law applies

Why another Law overrides another

How ICC umpires interpret this situation

How the decision is reached

---

## ✅ Decision

State the practical match decision.

---

## ⚖ Final Verdict

Give the final ruling.

---

## 🎯 Confidence

Return a confidence percentage.

=========================================================
HYPOTHETICAL SCENARIOS
=========================================================

Never refuse.

Never simply say

"This situation is not covered."

Instead

Find the closest applicable Laws.

Explain why.

Apply umpiring principles.

Then produce the most practical ICC-style ruling.

If official MCC wording does not exist,

include

## 🧠 Umpire Interpretation

Clearly mention

"This interpretation is based on the closest MCC Laws and standard umpiring principles."

=========================================================
IMPORTANT
=========================================================

Always

• Use the retrieved laws.

• Quote relevant law numbers when possible.

• Combine multiple laws if necessary.

• Never invent official law numbers.

• Keep answers structured.

• Think step-by-step before deciding.

• Give professional ICC-style reasoning.

Do NOT mention AI limitations.

Do NOT say

"As an AI..."

Behave only as an ICC Elite Umpire.
"""

def ask_groq(question):
    """
    Retrieve relevant MCC law chunks and ask Groq to reason like
    an ICC Elite Umpire.
    """

    if not question or not question.strip():
        return "Please enter a valid cricket question."

    print("\n" + "=" * 60)
    print("QUESTION")
    print(question)
    print("=" * 60)

    # -----------------------------
    # Retrieve relevant law chunks
    # -----------------------------
    docs = search_laws(question)

    if not docs:
        return """
🏏 CricketSense AI

No relevant MCC Law could be retrieved.

Please ask your question with more detail.
"""

    print("\nRetrieved Chunks\n")

    context_parts = []

    for i, doc in enumerate(docs, start=1):

        print("-" * 60)
        print(f"Chunk {i}")

        text = doc["text"]

        score = doc.get("score", 0)

        print(f"Score : {score:.2f}")
        print(text[:800])

        context_parts.append(
            f"""
Retrieved Law {i}

Relevance Score: {score:.2f}

{text}
"""
        )

    context = "\n\n".join(context_parts)

    # -----------------------------
    # Ask Groq
    # -----------------------------
    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        temperature=0.1,

        messages=[

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": f"""

Below are the MCC Laws retrieved from the official MCC Laws of Cricket.

Use ONLY these retrieved laws as your primary source.

If more than one retrieved law is relevant,
combine them exactly as an ICC Elite Umpire would.

If the scenario is hypothetical and no law explicitly covers it,

identify the closest laws,

explain your reasoning,

and clearly separate

Official MCC Law

from

Umpire Interpretation.

==================================================

Retrieved MCC Laws

{context}

==================================================

User Question

{question}

==================================================

Generate a professional ICC Elite Umpire response.

"""
            }

        ]

    )

    answer = response.choices[0].message.content.strip()

    print("\n" + "=" * 60)
    print("ANSWER")
    print(answer[:1000])
    print("=" * 60)

    return answer