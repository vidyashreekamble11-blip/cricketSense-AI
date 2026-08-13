import json
import os
import re
from rank_bm25 import BM25Okapi


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHUNKS_FILE = os.path.join(BASE_DIR, "mcc_chunks.json")

# Retrieve more Laws because hypothetical scenarios can involve
# multiple interacting Laws.
TOP_K = int(os.getenv("MCC_TOP_K", "6"))

# BM25 candidates before our second-stage scoring.
PREFILTER_N = int(os.getenv("MCC_PREFILTER_N", "40"))

DEBUG = os.getenv("MCC_DEBUG", "0") == "1"


# ============================================================
# LOAD MCC KNOWLEDGE
# ============================================================

if not os.path.exists(CHUNKS_FILE):
    raise FileNotFoundError(
        f"MCC chunks file not found:\n{CHUNKS_FILE}\n"
        "Run preprocess.py first."
    )

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

documents = [chunk.get("text", "") for chunk in chunks]


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """Convert text into searchable tokens."""
    if not text:
        return []

    text = text.lower()

    # Keep letters, numbers and hyphens.
    text = re.sub(r"[^a-z0-9\s-]", " ", text)

    return text.split()


tokenized_documents = [tokenize(doc) for doc in documents]

bm25 = BM25Okapi(tokenized_documents)


# ============================================================
# LAW KNOWLEDGE MAP
#
# This is used for:
#   1. scenario detection
#   2. query expansion
#   3. second-stage relevance scoring
#
# IMPORTANT:
# These keywords identify potentially relevant Laws.
# They do NOT directly determine the final umpire decision.
# ============================================================

LAW_KEYWORDS = {
    "1": [
        "players",
        "captain",
        "nomination of players",
        "team of eleven",
    ],

    "2": [
        "umpire",
        "umpires",
        "third umpire",
        "on-field umpire",
        "square leg umpire",
        "change of umpire",
        "fitness to continue",
    ],

    "3": [
        "scorer",
        "scorers",
        "scoring",
        "correctness of scores",
        "record runs",
    ],

    "4": [
        "the ball",
        "ball weight",
        "ball circumference",
        "replacement ball",
        "ball lost",
        "damaged ball",
        "ball splits",
        "two pieces",
        "split",
        "splits",
        "broken",
        "pieces",
        "unfit for play",
    ],

    "5": [
        "the bat",
        "bat dimensions",
        "width of bat",
        "edge of the bat",
    ],

    "6": [
        "the pitch",
        "pitch dimensions",
        "fitness of the pitch",
        "changing the pitch",
    ],

    "7": [
        "the creases",
        "popping crease",
        "bowling crease",
        "return crease",
    ],

    "8": [
        "the wickets",
        "stumps",
        "bails",
        "dimensions of the wicket",
    ],

    "9": [
        "preparation and maintenance",
        "rolling",
        "mowing",
        "watering",
        "footholes",
    ],

    "10": [
        "covering the pitch",
        "pitch covers",
    ],

    "11": [
        "intervals",
        "lunch interval",
        "tea interval",
        "drinks interval",
    ],

    "12": [
        "start of play",
        "cessation of play",
        "call of play",
        "call of time",
        "bad light",
        "suspension of play",
    ],

    "13": [
        "innings",
        "alternate innings",
        "forfeiture of an innings",
    ],

    "14": [
        "follow-on",
        "follow on",
    ],

    "15": [
        "declaration",
        "forfeiture",
    ],

    "16": [
        "the result",
        "match drawn",
        "win",
        "tie",
        "abandoned",
    ],

    "17": [
        "the over",
        "six balls",
        "over completed",
        "bowler changing ends",
    ],

    "18": [
        "scoring runs",
        "runs scored",
        "short run",
    ],

    "19": [
        "boundary",
        "boundaries",
        "four",
        "six",
        "boundary rope",
        "overthrow",
    ],

    "20": [
        "dead ball",
        "ball becomes dead",
        "umpire calls dead ball",
        "split",
        "splits",
        "broken",
        "unfit for play",
    ],

    "21": [
        "no ball",
        "no-ball",
        "illegal delivery",
        "front foot",
        "back foot",
        "over waist height",
        "over head height",
        "fielders encroaching",
    ],

    "22": [
        "wide ball",
        "wide",
    ],

    "23": [
        "bye",
        "byes",
        "leg bye",
        "leg byes",
    ],

    "24": [
        "fielders absence",
        "substitute",
        "substitutes",
        "runner",
    ],

    "25": [
        "batter's innings",
        "runner",
        "batter unable to run",
    ],

    "26": [
        "practice on the field",
        "trial run up",
        "practising",
    ],

    "27": [
        "wicketkeeper",
        "wicket-keeper",
        "keeper's gloves",
        "keeper standing up",
    ],

    "28": [
        "fielder",
        "fielders",
        "fielding position",
        "fielder's equipment",
    ],

    "29": [
        "wicket is down",
        "putting down the wicket",
        "bails removed",
    ],

    "30": [
        "batter out of ground",
        "batter's ground",
        "back into the ground",
    ],

    "31": [
        "appeal",
        "appealing",
        "how's that",
        "withdrawing an appeal",
    ],

    "32": [
        "bowled",
        "clean bowled",
        "ball hits the stumps",
    ],

    "33": [
        "caught",
        "catch",
        "fair catch",
        "held",
        "caught out",
        "one-handed catch",
    ],

    "34": [
        "hit the ball twice",
        "double hit",
        "second strike",
    ],

    "35": [
        "hit wicket",
        "batter dislodges the wicket",
        "own stroke breaks stumps",
    ],

    "36": [
        "lbw",
        "leg before wicket",
        "leg before",
        "pad",
        "impact",
        "pitched in line",
    ],

    "37": [
        "obstructing the field",
        "obstruction",
        "wilfully obstructs",
    ],

    "38": [
        "run out",
        "run-out",
        "wicket put down",
        "running between wickets",
    ],

    "39": [
        "stumped",
        "stumping",
        "keeper breaks the wicket",
    ],

    "40": [
        "timed out",
        "new batter arrival",
        "three minutes",
    ],

    "41": [
        "unfair play",
        "player's conduct",
        "ball tampering",
        "dangerous bowling",
        "time wasting",
        "deliberate distraction",
    ],
}


# ============================================================
# REVERSE KEYWORD INDEX
# ============================================================

KEYWORD_TO_LAWS = {}

for _law, _keywords in LAW_KEYWORDS.items():
    for _keyword in _keywords:
        KEYWORD_TO_LAWS.setdefault(_keyword, set()).add(_law)


# ============================================================
# LAW NUMBER DETECTION
# ============================================================

def extract_law_numbers(text):
    """
    Detect explicit references such as:

        Law 21
        Law No. 21

    Returns a set of strings.
    """
    if not text:
        return set()

    pattern = r"\bLaw\s+(?:No\.?\s*)?(\d{1,2})\b"

    return set(
        re.findall(
            pattern,
            text,
            re.IGNORECASE,
        )
    )


def detect_document_law(text):
    """
    Detect the Law number represented by a chunk.

    Priority:
        1. Explicit heading: LAW 33
        2. Subsection numbering such as 33.1

    This is only used to identify the source Law of a chunk.
    It does NOT decide the cricket outcome.
    """

    if not text:
        return None

    # --------------------------------------------------------
    # First try explicit heading.
    # Example:
    #     LAW 33
    #     LAW 33 - CAUGHT
    # --------------------------------------------------------

    heading_pattern = r"\bLAW\s+(\d{1,2})\b"

    matches = re.findall(
        heading_pattern,
        text,
        re.IGNORECASE,
    )

    if matches:
        return matches[0]

    # --------------------------------------------------------
    # Then try subsection numbering.
    #
    # Example:
    #     33.1
    #     33.2.1
    # --------------------------------------------------------

    subsection_pattern = r"(?<!\d)(\d{1,2})\.\d+(?:\.\d+)?\b"

    matches = re.findall(
        subsection_pattern,
        text,
    )

    if matches:
        law_number = matches[0]

        # Only accept actual Laws 1-41.
        try:
            if 1 <= int(law_number) <= 41:
                return law_number
        except ValueError:
            pass

    return None


# ============================================================
# BUILD LAW -> CHUNK INDEX
#
# This is the important retrieval improvement.
#
# BM25 can sometimes rank an unrelated chunk highly because
# common words such as "ball", "batter", "decision", etc.
# appear throughout the Laws.
#
# If our scenario detector identifies Law 33, for example,
# we make sure the actual Law 33 chunks are also considered
# during second-stage ranking.
# ============================================================

LAW_TO_CHUNK_INDICES = {}

for _idx, _chunk in enumerate(chunks):

    _text = _chunk.get("text", "")

    if not _text.strip():
        continue

    # Prefer the metadata if available.
    _law = str(
        _chunk.get("law_number", "")
    ).strip()

    # If metadata is empty, infer the Law from the text.
    if not _law:
        _law = detect_document_law(_text)

    if not _law:
        continue

    # Only index valid Laws.
    try:
        if not (1 <= int(_law) <= 41):
            continue
    except ValueError:
        continue

    LAW_TO_CHUNK_INDICES.setdefault(
        _law,
        set(),
    ).add(_idx)


# ============================================================
# SCENARIO DETECTION
# ============================================================

def _keyword_hits(question):
    """
    Detect Law keywords in the user's question.

    Multi-word keywords are handled token-by-token.

    Example:

        "splits into two equal pieces"

    can match:

        "two pieces"

    even though the words are not adjacent.
    """

    q_tokens = set(
        tokenize(question)
    )

    hits = []

    for keyword, laws in KEYWORD_TO_LAWS.items():

        keyword_tokens = set(
            tokenize(keyword)
        )

        if not keyword_tokens:
            continue

        if keyword_tokens.issubset(q_tokens):
            hits.append(
                (
                    keyword,
                    laws,
                )
            )

    return hits


def detect_scenario_laws(question):
    """
    Determine which Laws appear potentially relevant to the
    scenario.

    IMPORTANT:
    This does NOT determine the final decision.

    It only identifies Laws that should be considered.
    """

    detected = set()

    # Keyword-based detection.
    for _keyword, laws in _keyword_hits(question):
        detected.update(laws)

    # Explicit "Law 33" references.
    detected.update(
        extract_law_numbers(question)
    )

    return detected


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(question):
    """
    Expand the user's question with detected legal concepts
    so BM25 has stronger signals.
    """

    extra_terms = []

    # Add matched keywords.
    for keyword, laws in _keyword_hits(question):

        extra_terms.append(keyword)

        # Add explicit Law references.
        for law in laws:
            extra_terms.append(
                f"law {law}"
            )

    # Add Laws explicitly mentioned by the user.
    for law in extract_law_numbers(question):

        extra_terms.append(
            f"law {law}"
        )

    if not extra_terms:
        return question

    return (
        question
        + " "
        + " ".join(extra_terms)
    )


# ============================================================
# TEXT-LEVEL SCORING
# ============================================================

def important_term_score(question, document):
    """
    Small bonus for meaningful tokens shared by the question
    and the document.
    """

    query_tokens = set(
        tokenize(question)
    )

    document_tokens = set(
        tokenize(document)
    )

    matches = query_tokens & document_tokens

    return min(
        len(matches) * 0.20,
        2.0,
    )


def scenario_text_score(document, scenario_laws):
    """
    Score a document according to Laws detected from the
    scenario.

    Higher score means the document appears to belong to a
    potentially relevant Law.
    """

    if not scenario_laws:
        return 0.0

    document_lower = document.lower()

    score = 0.0

    for law in scenario_laws:

        # Strong signal: subsection belonging to the Law.
        if re.search(
            rf"\b{re.escape(str(law))}\.\d+(?:\.\d+)?\b",
            document_lower,
        ):
            score += 35.0

        # Keyword signals.
        for keyword in LAW_KEYWORDS.get(
            str(law),
            [],
        ):

            if keyword.lower() in document_lower:
                score += 6.0

    return min(
        score,
        60.0,
    )


def law_match_score(question, chunk):
    """
    Strong bonus when the user explicitly mentions
    a particular Law number.
    """

    question_laws = extract_law_numbers(
        question
    )

    if not question_laws:
        return 0.0

    metadata_law = str(
        chunk.get("law_number", "")
    ).strip()

    text_law = detect_document_law(
        chunk.get("text", "")
    )

    if (
        metadata_law in question_laws
        or text_law in question_laws
    ):
        return 8.0

    return 0.0


# Words that provide very little legal signal.
GENERIC_TERMS = {
    "question",
    "what",
    "which",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "does",
    "do",
    "can",
    "will",
    "would",
    "should",
    "decision",
}


def generic_word_count(question, document):
    """
    Count meaningful words shared between the question
    and the document.
    """

    query_tokens = (
        set(tokenize(question))
        - GENERIC_TERMS
    )

    document_tokens = set(
        tokenize(document)
    )

    return len(
        query_tokens & document_tokens
    )


# ============================================================
# DEDUPLICATION
# ============================================================

def _dedup_key(chunk):
    """
    Prevent repeated chunks from appearing in the final
    retrieval results.
    """

    text = chunk.get(
        "text",
        "",
    ).strip().lower()

    return (
        str(
            chunk.get(
                "law_number",
                "",
            )
        ),
        text[:80],
    )


# ============================================================
# SEARCH MCC LAWS
# ============================================================

def search_laws(question, top_k=None):
    """
    Retrieve the most relevant MCC Law passages.

    Retrieval has two stages:

        Stage 1:
            BM25 finds linguistically relevant chunks.

        Stage 2:
            Detected Law chunks are forcibly added to the
            candidate pool and then all candidates are
            re-ranked using legal/scenario signals.

    Returns:

        [
            {
                "law_number": ...,
                "page": ...,
                "text": ...,
                "score": ...
            }
        ]

    IMPORTANT:
    This function retrieves evidence only.
    It does NOT decide whether the batter is out.
    """

    top_k = (
        top_k
        if top_k is not None
        else TOP_K
    )

    if not question or not question.strip():
        return []

    # --------------------------------------------------------
    # Detect potentially relevant Laws BEFORE retrieval.
    # --------------------------------------------------------

    scenario_laws = detect_scenario_laws(
        question
    )

    # --------------------------------------------------------
    # Expand the query for BM25.
    # --------------------------------------------------------

    expanded_query = expand_query(
        question
    )

    query_tokens = tokenize(
        expanded_query
    )

    # --------------------------------------------------------
    # Stage 1:
    #
    # BM25 over the complete MCC corpus.
    # --------------------------------------------------------

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:PREFILTER_N]

    # --------------------------------------------------------
    # Stage 2:
    #
    # Start with BM25 candidates.
    # Then add ALL chunks belonging to Laws detected from
    # the scenario.
    #
    # This prevents BM25 from accidentally eliminating the
    # relevant Law before our custom scoring runs.
    # --------------------------------------------------------

    candidate_indices = set(
        bm25_ranked_indices
    )

    for law in scenario_laws:

        law_indices = LAW_TO_CHUNK_INDICES.get(
            str(law),
            set(),
        )

        candidate_indices.update(
            law_indices
        )

    # --------------------------------------------------------
    # Safety fallback:
    #
    # If no Laws were detected and BM25 somehow returns
    # nothing, consider the top BM25 documents.
    # --------------------------------------------------------

    if not candidate_indices:
        candidate_indices.update(
            bm25_ranked_indices
        )

    # --------------------------------------------------------
    # Second-stage scoring.
    # --------------------------------------------------------

    candidates = []

    seen = set()

    for idx in candidate_indices:

        chunk = chunks[idx]

        text = chunk.get(
            "text",
            "",
        )

        if not text.strip():
            continue

        key = _dedup_key(
            chunk
        )

        if key in seen:
            continue

        seen.add(key)

        bm25_score = float(
            bm25_scores[idx]
        )

        important_score = (
            important_term_score(
                question,
                text,
            )
        )

        law_score = (
            law_match_score(
                question,
                chunk,
            )
        )

        scenario_score = (
            scenario_text_score(
                text,
                scenario_laws,
            )
        )

        direct_score = min(
            generic_word_count(
                question,
                text,
            ) * 0.15,
            1.5,
        )

        metadata_law = str(
            chunk.get(
                "law_number",
                "",
            )
        ).strip()

        text_law = detect_document_law(
            text
        )

        actual_law = (
            metadata_law
            if metadata_law
            else text_law
        )

        # If metadata is missing but text detection succeeded.
        if not actual_law:
            actual_law = text_law

        final_score = (
            bm25_score
            + important_score
            + law_score
            + scenario_score
            + direct_score
        )

        candidates.append(
            {
                "score": round(
                    final_score,
                    3,
                ),
                "law_number": actual_law,
                "page": chunk.get(
                    "page"
                ),
                "text": text.strip(),
            }
        )

    # --------------------------------------------------------
    # Final ranking.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    results = candidates[:top_k]

    # --------------------------------------------------------
    # DEBUG OUTPUT
    # --------------------------------------------------------

    if DEBUG:

        print(
            "\n"
            + "=" * 60
        )

        print(
            "MCC RETRIEVER"
        )

        print(
            "=" * 60
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Detected scenario laws: "
            f"{sorted(scenario_laws, key=lambda x: int(x))}"
        )

        print(
            f"BM25 candidates: "
            f"{len(bm25_ranked_indices)}"
        )

        print(
            f"Combined candidates: "
            f"{len(candidate_indices)}"
        )

        print(
            f"Total MCC chunks: "
            f"{len(chunks)}"
        )

        print(
            "-" * 60
        )

        for i, result in enumerate(
            results,
            start=1,
        ):

            preview = (
                result["text"]
                .replace(
                    "\n",
                    " ",
                )[:180]
            )

            print(
                f"[{i}] "
                f"Law={result['law_number']} "
                f"Page={result['page']} "
                f"Score={result['score']}"
            )

            print(
                f"    {preview}..."
            )

        print(
            "=" * 60
        )

    return results