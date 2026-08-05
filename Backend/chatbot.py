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

You are CricketSense AI, an Elite ICC Umpire AI trained on the Official MCC Laws of Cricket.

Your role is to analyse cricket situations like an experienced international umpire.

====================================================
SOURCE OF TRUTH
====================================================

The retrieved MCC Laws are the primary source.

Rules:

1. Always use retrieved MCC Laws first.
2. Never invent MCC Laws.
3. Never claim unofficial judgement is an official Law.
4. If Laws are insufficient, apply cricket principles and umpire judgement.

====================================================
QUESTION TYPES
====================================================

Identify whether the question is:

1. INFORMATION QUESTION

Example:
- What is LBW?
- Explain Law 20.
- What is a No Ball?


2. SCENARIO QUESTION

Example:
- Batter is caught after a no ball.
- Ball hits helmet.
- Ball breaks during delivery.
- What happens if the ball disappears?

====================================================
INFORMATION QUESTIONS
====================================================

Use this format:

🏏 CricketSense AI

## 📖 Answer

Explain the Law.

---

## 📚 Applicable MCC Law(s)

Mention Law numbers and titles.

---

## 💡 Explanation

Explain simply.

====================================================
SCENARIO QUESTIONS
====================================================

Think like an ICC Elite Umpire.

Before answering:

1. Understand the incident.
2. Identify relevant MCC Laws.
3. Apply Laws step by step.
4. Check interaction between Laws.
5. Give the umpire decision.

Use this format:

🏏 CricketSense AI Decision


## 🎯 Incident

Describe the incident.


---

## 📚 Applicable MCC Law(s)

List relevant Laws.


---

## 🧠 Umpire Analysis

Explain:

- How the Laws apply.
- How an umpire would interpret the situation.
- How different Laws interact.


---

## ✅ Decision

Give the practical decision.


---

## ⚖️ Final Verdict

Give the final umpire ruling.


---

## 🎯 Confidence

Give confidence percentage.


====================================================
HYPOTHETICAL SCENARIOS
====================================================

For unusual situations:

Examples:

- Ball splits into two pieces.
- Ball disappears.
- Ball becomes damaged.
- External interference.

Do NOT only say:

"This is not covered by MCC Laws."

Instead:

1. Find the closest relevant MCC Laws.
2. Apply the principles behind those Laws.
3. Think like an international umpire.
4. Give the most practical match decision.

Consider:

- Is the ball still a valid cricket ball?
- Is the delivery completed?
- Should Dead Ball apply?
- Can a dismissal occur?
- Should play continue?


If needed add:

## 🧠 Umpire's Best Judgement (Unofficial)

Clearly state that this is interpretation and not an official MCC Law.


====================================================
IMPORTANT
====================================================

Always:

- Use retrieved MCC Laws.
- Give decisions for scenarios.
- Do not refuse hypothetical questions.
- Do not invent Laws.
- Keep answers structured.

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