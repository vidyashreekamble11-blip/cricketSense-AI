import json
import os
import re
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_FILE = os.path.join(BASE_DIR, "mcc_chunks.json")

# Load chunks once
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

documents = [chunk["text"] for chunk in chunks]


def tokenize(text):
    """
    Clean and tokenize text.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


tokenized_documents = [tokenize(doc) for doc in documents]

bm25 = BM25Okapi(tokenized_documents)


def expand_query(question):
    """
    Cricket-aware query expansion.
    """

    q = question.lower()

    extra = []

    mapping = {

        "lbw": [
            "leg before wicket",
            "wicket",
            "dismissal"
        ],

        "no ball": [
            "illegal delivery",
            "law 21",
            "dismissal"
        ],

        "wide": [
            "law 22",
            "wide ball"
        ],

        "catch": [
            "caught",
            "fielder",
            "fair catch"
        ],

        "helmet": [
            "protective helmet",
            "dead ball",
            "fielder",
            "keeper"
        ],

        "dead ball": [
            "law 20"
        ],

        "boundary": [
            "four",
            "six",
            "rope"
        ],

        "ball damaged": [
            "damaged ball",
            "replace ball"
        ],

        "appeal": [
            "law 31"
        ],

        "run out": [
            "dismissal",
            "wicket"
        ]

    }

    for key in mapping:

        if key in q:

            extra.extend(mapping[key])

    return question + " " + " ".join(extra)


def search_laws(question, top_k=10):

    expanded = expand_query(question)

    tokens = tokenize(expanded)

    scores = bm25.get_scores(tokens)

    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )

    results = []

    for score, chunk in ranked[:top_k]:

        results.append({

            "score": float(score),

            "text": chunk["text"]

        })

    return results