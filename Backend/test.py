
import os
import re
import json
from collections import Counter

from retriever import search_laws


# ============================================================
# CRICKETSENSE-AI RETRIEVER TEST
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CHUNKS_FILE = os.path.join(
    BASE_DIR,
    "mcc_chunks.json"
)

EXPECTED_LAWS = set(range(1, 43))

TOP_K = 10

# Maximum characters displayed from each retrieved chunk
PREVIEW_LIMIT = 250


# ============================================================
# MCC LAW TITLES
# ============================================================

LAW_TITLES = {
    1: "The players",
    2: "The umpires",
    3: "The scorers",
    4: "The ball",
    5: "The bat",
    6: "The pitch",
    7: "The creases",
    8: "The wickets",
    9: "Preparation and maintenance of the playing area",
    10: "Covering the pitch",
    11: "Intervals",
    12: "Start of play; cessation of play",
    13: "Innings",
    14: "The follow-on",
    15: "Declaration and forfeiture",
    16: "The result",
    17: "The over",
    18: "Scoring runs",
    19: "Boundaries",
    20: "Dead ball",
    21: "No ball",
    22: "Wide ball",
    23: "Bye and Leg bye",
    24: "Fielder’s absence; substitutes",
    25: "Batter’s innings; runners",
    26: "Practice on the field",
    27: "The wicket-keeper",
    28: "The fielder",
    29: "The wicket is down",
    30: "Batter out of his/her ground",
    31: "Appeals",
    32: "Bowled",
    33: "Caught",
    34: "Hit the ball twice",
    35: "Hit wicket",
    36: "Leg before wicket",
    37: "Obstructing the field",
    38: "Run out",
    39: "Stumped",
    40: "Timed out",
    41: "Unfair play",
    42: "Players’ conduct",
}


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():
    """
    Load the generated MCC chunk JSON.
    """

    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(
            f"\nMCC chunk file not found:\n{CHUNKS_FILE}\n"
            f"\nRun preprocess.py first.\n"
        )

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError(
            "mcc_chunks.json must contain a JSON list."
        )

    return chunks


# ============================================================
# DETECT LAW HEADINGS INSIDE CHUNK
# ============================================================

def detect_headings(text):
    """
    Detect genuine MCC Law headings inside a chunk.

    Examples detected:

        LAW 33 CAUGHT
        LAW 41 UNFAIR PLAY
        LAW 42 PLAYERS' CONDUCT

    References such as:

        Law 18.11
        Law 41.6

    are NOT treated as Law headings.
    """

    if not text:
        return []

    headings = []

    lines = text.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # ----------------------------------------------------
        # Remove PDF page header if present
        # ----------------------------------------------------

        line = re.sub(
            r"^Laws of Cricket.*?\b\d{1,3}\s*$",
            "",
            line,
            flags=re.IGNORECASE
        ).strip()

        # ----------------------------------------------------
        # Normal heading
        #
        # LAW 33 CAUGHT
        # LAW 41 UNFAIR PLAY
        # ----------------------------------------------------

        match = re.match(
            r"^LAW\s+(\d{1,2})\s+(.+?)\s*$",
            line,
            flags=re.IGNORECASE
        )

        if match:

            law_number = int(
                match.group(1)
            )

            if 1 <= law_number <= 42:

                headings.append(
                    law_number
                )

            continue

        # ----------------------------------------------------
        # Heading may be extracted as:
        #
        # LAW 33
        # CAUGHT
        # ----------------------------------------------------

        match = re.match(
            r"^LAW\s+(\d{1,2})\s*$",
            line,
            flags=re.IGNORECASE
        )

        if match:

            law_number = int(
                match.group(1)
            )

            if 1 <= law_number <= 42:

                headings.append(
                    law_number
                )

    return sorted(
        set(headings)
    )


# ============================================================
# CHUNK INTEGRITY CHECK
# ============================================================

def check_chunk_integrity(chunks):
    """
    Check whether a chunk assigned to one Law contains
    another genuine Law heading.

    Example:

        Assigned Law = 26

        Chunk contains:
        LAW 41 UNFAIR PLAY

    This is suspicious because the chunk crosses a Law
    boundary.
    """

    suspicious = []

    for chunk in chunks:

        assigned_law = chunk.get(
            "law_number"
        )

        text = chunk.get(
            "text",
            ""
        )

        if assigned_law is None:
            continue

        try:
            assigned_law = int(
                assigned_law
            )
        except (
            ValueError,
            TypeError
        ):
            continue

        headings = detect_headings(
            text
        )

        # ----------------------------------------------------
        # A chunk can contain its own Law heading.
        # That is valid.
        #
        # Any other Law heading is suspicious.
        # ----------------------------------------------------

        unexpected_laws = [
            law
            for law in headings
            if law != assigned_law
        ]

        if unexpected_laws:

            suspicious.append(
                {
                    "id": chunk.get("id"),
                    "page": chunk.get("page"),
                    "assigned_law": assigned_law,
                    "unexpected_laws": unexpected_laws,
                    "text": text
                }
            )

    return suspicious


# ============================================================
# LAW DISTRIBUTION
# ============================================================

def show_distribution(chunks):
    """
    Display the number of chunks belonging to each Law.
    """

    distribution = Counter()

    for chunk in chunks:

        law = chunk.get(
            "law_number"
        )

        try:
            law = int(law)
        except (
            ValueError,
            TypeError
        ):
            continue

        distribution[law] += 1

    print()
    print("=" * 70)
    print("LAW DISTRIBUTION")
    print("=" * 70)
    print()

    for law in sorted(distribution):

        print(
            f"Law {law:2d}: "
            f"{distribution[law]} chunks"
        )

    return distribution


# ============================================================
# LAW COVERAGE CHECK
# ============================================================

def check_law_coverage(chunks):
    """
    Verify that all 42 MCC Laws exist in the chunk database.
    """

    detected_laws = set()

    for chunk in chunks:

        law = chunk.get(
            "law_number"
        )

        try:
            law = int(law)
        except (
            ValueError,
            TypeError
        ):
            continue

        detected_laws.add(
            law
        )

    missing = sorted(
        EXPECTED_LAWS - detected_laws
    )

    unexpected = sorted(
        detected_laws - EXPECTED_LAWS
    )

    print()
    print("=" * 70)
    print("LAW COVERAGE VALIDATION")
    print("=" * 70)
    print()

    print(
        f"Expected Laws : {len(EXPECTED_LAWS)}"
    )

    print(
        f"Detected Laws : {len(detected_laws)}"
    )

    if missing:

        print()
        print(
            "MISSING LAWS:"
        )

        print(
            ", ".join(
                str(law)
                for law in missing
            )
        )

    if unexpected:

        print()
        print(
            "UNEXPECTED LAWS:"
        )

        print(
            ", ".join(
                str(law)
                for law in unexpected
            )
        )

    if not missing and not unexpected:

        print()
        print(
            "SUCCESS: All 42 MCC Laws are present."
        )

    return (
        detected_laws,
        missing,
        unexpected
    )


# ============================================================
# RETRIEVER TEST
# ============================================================

def run_retriever_test(
    question,
    expected_law
):
    """
    Run one retrieval test.

    The test checks whether the expected Law appears
    in the retrieved results.
    """

    print()
    print("#" * 70)
    print("QUERY:")
    print(question)
    print(
        f"Expected Law: {expected_law}"
    )
    print("#" * 70)

    try:

        results = search_laws(
            question,
            top_k=TOP_K
        )

    except Exception as error:

        print()
        print(
            "ERROR while running retriever:"
        )

        print(
            repr(error)
        )

        return False

    print()
    print(
        f"Retrieved {len(results)} "
        f"MCC Law chunks."
    )

    if not results:

        print(
            "WARNING: No retrieval results."
        )

        return False

    expected_found = False

    for index, result in enumerate(
        results,
        start=1
    ):

        law = result.get(
            "law_number"
        )

        page = result.get(
            "page"
        )

        score = result.get(
            "score",
            0
        )

        text = result.get(
            "text",
            ""
        )

        try:
            law_int = int(law)
        except (
            ValueError,
            TypeError
        ):
            law_int = None

        if law_int == expected_law:

            expected_found = True

        # ----------------------------------------------------
        # Check whether retrieved result itself is mixed.
        # ----------------------------------------------------

        headings = detect_headings(
            text
        )

        unexpected = [
            h
            for h in headings
            if h != law_int
        ]

        if unexpected:

            integrity = (
                f"WARNING - mixed headings "
                f"{unexpected}"
            )

        else:

            integrity = "OK"

        print()
        print("=" * 70)

        print(
            f"RESULT {index}"
        )

        print(
            f"Law   : {law}"
        )

        print(
            f"Page  : {page}"
        )

        print(
            f"Score : {score}"
        )

        print(
            f"Integrity: {integrity}"
        )

        print()
        print("TEXT:")

        print()

        print(
            text[:PREVIEW_LIMIT]
        )

        if len(text) > PREVIEW_LIMIT:

            print(
                "\n...[text truncated]..."
            )

    print()

    if expected_found:

        print(
            f"SUCCESS: Expected Law "
            f"{expected_law} was retrieved."
        )

    else:

        print(
            f"WARNING: Expected Law "
            f"{expected_law} was NOT found "
            f"in the top {TOP_K} results."
        )

    return expected_found


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_CASES = [

    (
        "When is a batter out Caught and what are "
        "the conditions for a fair catch?",
        33
    ),

    (
        "When can a batter be dismissed for "
        "Obstructing the field?",
        37
    ),

    (
        "When does the ball become dead?",
        20
    ),

    (
        "What is a No ball and when should the "
        "umpire call it?",
        21
    ),

    (
        "How many valid balls are there in an over?",
        17
    ),

    (
        "When is a batter out Run out?",
        38
    ),

    (
        "When is a batter out Stumped?",
        39
    ),

    (
        "What are the conditions for a batter to be "
        "given out Leg before wicket?",
        36
    ),

    (
        "When is a batter out Bowled?",
        32
    ),

    (
        "What are the rules for a fair catch?",
        33
    ),

]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CRICKETSENSE-AI RETRIEVER TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    try:

        chunks = load_chunks()

    except Exception as error:

        print()
        print(
            "ERROR:"
        )

        print(
            error
        )

        return

    print()
    print(
        f"Loaded {len(chunks)} "
        f"MCC Law chunks."
    )

    # --------------------------------------------------------
    # Basic chunk validation
    # --------------------------------------------------------

    invalid_chunks = []

    for chunk in chunks:

        required_fields = {
            "id",
            "page",
            "law_number",
            "text"
        }

        missing_fields = (
            required_fields
            - set(chunk.keys())
        )

        if missing_fields:

            invalid_chunks.append(
                (
                    chunk.get("id"),
                    missing_fields
                )
            )

    if invalid_chunks:

        print()
        print(
            f"WARNING: "
            f"{len(invalid_chunks)} chunks "
            f"have missing fields."
        )

    else:

        print(
            "Chunk structure: OK"
        )

    # --------------------------------------------------------
    # Run retrieval tests
    # --------------------------------------------------------

    passed_tests = 0

    failed_tests = 0

    for question, expected_law in TEST_CASES:

        success = run_retriever_test(
            question,
            expected_law
        )

        if success:

            passed_tests += 1

        else:

            failed_tests += 1

    # --------------------------------------------------------
    # Global integrity
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GLOBAL MCC CHUNK INTEGRITY CHECK")
    print("=" * 70)

    suspicious = check_chunk_integrity(
        chunks
    )

    print()

    print(
        f"Total chunks checked : "
        f"{len(chunks)}"
    )

    print(
        f"Suspicious chunks    : "
        f"{len(suspicious)}"
    )

    if suspicious:

        print()
        print(
            "WARNING: Some chunks contain "
            "another Law heading."
        )

        print()

        for item in suspicious:

            print(
                f"Chunk {item['id']} | "
                f"Page {item['page']} | "
                f"Assigned Law "
                f"{item['assigned_law']} | "
                f"Unexpected Laws "
                f"{item['unexpected_laws']}"
            )

    else:

        print()
        print(
            "SUCCESS: No mixed Law headings detected."
        )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    distribution = show_distribution(
        chunks
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    (
        detected_laws,
        missing_laws,
        unexpected_laws
    ) = check_law_coverage(
        chunks
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RETRIEVER TEST SUMMARY")
    print("=" * 70)

    print()

    print(
        f"Retrieval tests passed : "
        f"{passed_tests}/{len(TEST_CASES)}"
    )

    print(
        f"Retrieval tests failed : "
        f"{failed_tests}/{len(TEST_CASES)}"
    )

    print(
        f"Total chunks           : "
        f"{len(chunks)}"
    )

    print(
        f"Laws detected          : "
        f"{len(detected_laws)}/42"
    )

    print(
        f"Mixed chunks           : "
        f"{len(suspicious)}"
    )

    print()

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if (
        passed_tests == len(TEST_CASES)
        and not missing_laws
        and not unexpected_laws
        and not suspicious
    ):

        print(
            "SUCCESS: MCC RETRIEVER "
            "PASSED ALL TESTS."
        )

        print()
        print(
            "The MCC knowledge base is ready "
            "for the next RAG/chatbot stage."
        )

    elif (
        passed_tests == len(TEST_CASES)
        and not missing_laws
        and not unexpected_laws
    ):

        print(
            "RETRIEVER FUNCTIONALITY: PASS"
        )

        print(
            "However, mixed Law chunks were detected."
        )

        print(
            "Run preprocess.py again with the "
            "corrected Law-boundary chunking."
        )

    else:

        print(
            "WARNING: RETRIEVER TESTS NEED ATTENTION."
        )

        if failed_tests:

            print(
                "- Some expected Laws were not "
                "retrieved."
            )

        if missing_laws:

            print(
                "- Some MCC Laws are missing "
                "from the chunk database."
            )

        if unexpected_laws:

            print(
                "- Unexpected Law numbers were found."
            )

    print()
    print("=" * 70)
    print("RETRIEVER TEST COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

