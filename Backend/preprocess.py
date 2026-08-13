```python
import os
import re
import json
from collections import Counter
from pypdf import PdfReader


# ============================================================
# CRICKETSENSE-AI
# MCC LAWS KNOWLEDGE BUILDER
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PDF_PATH = os.path.join(
    BASE_DIR,
    "knowledge",
    "Laws-of-Cricket-2017-Code-3rd-Edition-2022_1.pdf"
)

OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mcc_chunks.json"
)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 1800
OVERLAP = 250

# Table of Contents pages
TOC_PAGES = {52, 77, 78, 79}

# MCC Laws 1-42
EXPECTED_LAWS = set(range(1, 43))


# ============================================================
# LAW TITLES
# ============================================================

LAW_TITLES = {
    1: "THE PLAYERS",
    2: "THE UMPIRES",
    3: "THE SCORERS",
    4: "THE BALL",
    5: "THE BAT",
    6: "THE PITCH",
    7: "THE CREASES",
    8: "THE WICKETS",
    9: "PREPARATION AND MAINTENANCE OF THE PLAYING AREA",
    10: "COVERING THE PITCH",
    11: "INTERVALS",
    12: "START OF PLAY; CESSATION OF PLAY",
    13: "INNINGS",
    14: "THE FOLLOW-ON",
    15: "DECLARATION AND FORFEITURE",
    16: "THE RESULT",
    17: "THE OVER",
    18: "SCORING RUNS",
    19: "BOUNDARIES",
    20: "DEAD BALL",
    21: "NO BALL",
    22: "WIDE BALL",
    23: "BYE AND LEG BYE",
    24: "FIELDER'S ABSENCE; SUBSTITUTES",
    25: "BATTER'S INNINGS; RUNNERS",
    26: "PRACTICE ON THE FIELD",
    27: "THE WICKET-KEEPER",
    28: "THE FIELDER",
    29: "THE WICKET IS DOWN",
    30: "BATTER OUT OF HIS/HER GROUND",
    31: "APPEALS",
    32: "BOWLED",
    33: "CAUGHT",
    34: "HIT THE BALL TWICE",
    35: "HIT WICKET",
    36: "LEG BEFORE WICKET",
    37: "OBSTRUCTING THE FIELD",
    38: "RUN OUT",
    39: "STUMPED",
    40: "TIMED OUT",
    41: "UNFAIR PLAY",
    42: "PLAYERS' CONDUCT",
}


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    """
    Clean PDF extracted text while preserving useful line
    boundaries.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize non-breaking spaces
    text = text.replace("\xa0", " ")

    # Normalize tabs/spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# NORMALIZE LINE
# ============================================================

def normalize_line(line):
    """
    Normalize a single PDF-extracted line for heading detection.
    """

    if not line:
        return ""

    line = line.replace("\xa0", " ")
    line = re.sub(r"[ \t]+", " ", line)

    return line.strip()


# ============================================================
# REMOVE PDF PAGE HEADER
# ============================================================

def remove_page_header(line):
    """
    Remove MCC PDF page header such as:

    Laws of Cricket 2017 Code (3rd Edition - 2022) 47
    """

    if not line:
        return ""

    pattern = (
        r"^Laws\s+of\s+Cricket\s+2017\s+Code"
        r".*?\d{1,3}\s*$"
    )

    if re.match(
        pattern,
        line,
        flags=re.IGNORECASE
    ):
        return ""

    return line


# ============================================================
# CHECK WHETHER TEXT LOOKS LIKE A LAW TITLE
# ============================================================

def title_matches_expected(law_number, title):
    """
    Check the detected heading against known MCC Law titles.

    This prevents ordinary text such as:

        Law 38.3
        Law 41.3.2

    from being treated as a new main Law.
    """

    if law_number not in LAW_TITLES:
        return False

    if not title:
        return False

    normalized_title = re.sub(
        r"[^a-z0-9]+",
        " ",
        title.lower()
    ).strip()

    expected_title = re.sub(
        r"[^a-z0-9]+",
        " ",
        LAW_TITLES[law_number].lower()
    ).strip()

    # Exact match
    if normalized_title == expected_title:
        return True

    # PDF extraction may truncate a long title.
    # Accept if the expected title starts with the extracted title
    # and the extracted title is reasonably long.
    if (
        len(normalized_title) >= 12
        and expected_title.startswith(normalized_title)
    ):
        return True

    return False


# ============================================================
# FIND MCC LAW HEADINGS
# ============================================================

def find_law_headings(text):
    """
    Detect genuine main MCC Law headings.

    IMPORTANT:

    This detects:

        LAW 38 RUN OUT
        LAW 39 STUMPED
        LAW 40 TIMED OUT
        LAW 41 UNFAIR PLAY
        LAW 42 PLAYERS' CONDUCT

    It does NOT detect:

        Law 38.1
        Law 38.2
        Law 41.3.2
        Law 41.6

    as new Law sections.
    """

    if not text:
        return []

    lines = text.splitlines()

    headings = []

    i = 0

    while i < len(lines):

        raw_line = lines[i]

        line = normalize_line(raw_line)

        line = remove_page_header(line)

        if not line:
            i += 1
            continue

        # ----------------------------------------------------
        # CASE 1:
        #
        # LAW 38 RUN OUT
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

            title = match.group(2).strip()

            # Main Laws only
            if 1 <= law_number <= 42:

                # Reject subsection-like text
                if not re.search(
                    r"\d+\.\d+",
                    title
                ):

                    if title_matches_expected(
                        law_number,
                        title
                    ):

                        headings.append(
                            {
                                "law_number": law_number,
                                "title": title,
                                "line_index": i
                            }
                        )

            i += 1
            continue

        # ----------------------------------------------------
        # CASE 2:
        #
        # LAW 38
        # RUN OUT
        #
        # Some PDF extraction can split the heading.
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

                # Search next non-empty line
                j = i + 1

                while j < len(lines):

                    next_line = normalize_line(
                        lines[j]
                    )

                    next_line = remove_page_header(
                        next_line
                    )

                    if next_line:
                        break

                    j += 1

                if j < len(lines):

                    title = next_line

                    # Must look like a real title
                    if title_matches_expected(
                        law_number,
                        title
                    ):

                        headings.append(
                            {
                                "law_number": law_number,
                                "title": title,
                                "line_index": i
                            }
                        )

                        i = j
                        continue

        i += 1

    return headings


# ============================================================
# REMOVE DUPLICATE HEADINGS
# ============================================================

def remove_duplicate_headings(headings):
    """
    Remove duplicate heading detections on the same page.
    """

    result = []

    seen = set()

    for heading in headings:

        key = (
            heading["law_number"],
            heading["line_index"]
        )

        if key in seen:
            continue

        seen.add(key)

        result.append(heading)

    return result


# ============================================================
# BUILD GLOBAL LAW BOUNDARIES
# ============================================================

def build_global_law_boundaries(reader):
    """
    Build a GLOBAL sequence of all 42 Law boundaries.

    Instead of deciding the Law independently on every page,
    this creates a continuous document-level structure.

    Each heading receives:

        page
        law_number
        title
        line_index
    """

    boundaries = []

    for page_no, page in enumerate(
        reader.pages,
        start=1
    ):

        if page_no in TOC_PAGES:

            print(
                f"Skipping TOC page: {page_no}"
            )

            continue

        text = page.extract_text()

        if not text:
            continue

        text = clean_text(text)

        if not text:
            continue

        headings = find_law_headings(text)

        headings = remove_duplicate_headings(
            headings
        )

        for heading in headings:

            boundaries.append(
                {
                    "page": page_no,
                    "law_number": heading["law_number"],
                    "title": heading["title"],
                    "line_index": heading["line_index"]
                }
            )

    return boundaries


# ============================================================
# VALIDATE LAW BOUNDARIES
# ============================================================

def validate_boundaries(boundaries):
    """
    Validate that the document contains all 42 Laws
    in the expected order.
    """

    detected = []

    for boundary in boundaries:

        law_number = boundary["law_number"]

        if law_number not in detected:

            detected.append(
                law_number
            )

    print()
    print(
        f"Law sections detected: "
        f"{len(detected)}"
    )

    print()
    print("Detected MCC Laws:")
    print()

    for law_number in detected:

        title = LAW_TITLES.get(
            law_number,
            "UNKNOWN"
        )

        print(
            f"Law {law_number:2d} – "
            f"{title.title()}"
        )

    missing = sorted(
        EXPECTED_LAWS - set(detected)
    )

    unexpected = sorted(
        set(detected) - EXPECTED_LAWS
    )

    print()
    print(
        f"Detected laws: {len(detected)}"
    )

    if detected:

        print(
            f"Detected range: "
            f"{min(detected)} - "
            f"{max(detected)}"
        )

    if missing:

        print()
        print(
            "WARNING - Missing laws:"
        )

        print(
            ", ".join(
                str(x)
                for x in missing
            )
        )

    elif unexpected:

        print()
        print(
            "WARNING - Unexpected laws:"
        )

        print(
            ", ".join(
                str(x)
                for x in unexpected
            )
        )

    else:

        expected_order = list(
            range(1, 43)
        )

        if detected == expected_order:

            print()
            print(
                "SUCCESS - All 42 MCC Laws detected "
                "in correct order."
            )

        else:

            print()
            print(
                "WARNING - All Laws may be present "
                "but ordering is not 1-42."
            )

    return detected


# ============================================================
# SPLIT TEXT INTO CHUNKS
# ============================================================

def split_large_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP
):
    """
    Split text into approximately CHUNK_SIZE characters.

    Chunk boundaries prefer paragraphs and lines.
    """

    if not text:
        return []

    text = text.strip()

    if len(text) <= chunk_size:

        return [text]

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        if end < text_length:

            # Prefer paragraph boundary
            paragraph_end = text.rfind(
                "\n\n",
                start,
                end
            )

            if paragraph_end > start + 500:

                end = paragraph_end

            else:

                # Prefer line boundary
                line_end = text.rfind(
                    "\n",
                    start,
                    end
                )

                if line_end > start + 500:

                    end = line_end

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end >= text_length:

            break

        next_start = end - overlap

        if next_start <= start:

            next_start = end

        start = next_start

    return chunks


# ============================================================
# PROCESS ONE PAGE
# ============================================================

def extract_page_sections(
    text,
    page_no,
    current_law
):
    """
    Split one page into sections using ONLY genuine
    MCC Law headings.

    Example:

        Page 53

        LAW 38 RUN OUT
        ...
        LAW 39 STUMPED
        ...

    becomes:

        Law 38 -> text before Law 39
        Law 39 -> remaining text

    Text before the first heading belongs to the previous
    Law, but ONLY if a valid current Law already exists.
    """

    lines = text.splitlines()

    headings = find_law_headings(text)

    headings = remove_duplicate_headings(
        headings
    )

    sections = []

    # --------------------------------------------------------
    # No heading on this page
    # --------------------------------------------------------

    if not headings:

        if current_law is not None:

            sections.append(
                {
                    "law_number": current_law,
                    "text": text
                }
            )

        return sections, current_law

    # --------------------------------------------------------
    # Text before first heading
    # --------------------------------------------------------

    first_heading_line = headings[0][
        "line_index"
    ]

    if first_heading_line > 0:

        before_text = "\n".join(
            lines[
                0:first_heading_line
            ]
        ).strip()

        if before_text and current_law is not None:

            sections.append(
                {
                    "law_number": current_law,
                    "text": before_text
                }
            )

    # --------------------------------------------------------
    # Process each heading
    # --------------------------------------------------------

    for index, heading in enumerate(
        headings
    ):

        law_number = heading[
            "law_number"
        ]

        start_line = heading[
            "line_index"
        ]

        if index + 1 < len(headings):

            end_line = headings[
                index + 1
            ]["line_index"]

        else:

            end_line = len(lines)

        section_text = "\n".join(
            lines[
                start_line:end_line
            ]
        ).strip()

        if section_text:

            sections.append(
                {
                    "law_number": law_number,
                    "text": section_text
                }
            )

        current_law = law_number

    return sections, current_law


# ============================================================
# CHECK FOR CROSS-LAW CONTAMINATION
# ============================================================

def detect_cross_law_contamination(data):
    """
    Check whether a chunk assigned to one Law contains a
    strong indication that it belongs to another Law.

    This is a safety check, not a replacement for heading
    detection.

    Example:

        Assigned Law 38

        Text:
        41.3.2 It is an offence for any player...

    is suspicious.
    """

    problems = []

    for item in data:

        assigned_law = item[
            "law_number"
        ]

        text = item[
            "text"
        ]

        # Find references that look like actual section
        # numbering at the beginning of a line.
        references = re.findall(
            r"(?m)^\s*(\d{1,2})\.(\d+)(?:\.\d+)?\b",
            text
        )

        for law_str, subsection in references:

            referenced_law = int(
                law_str
            )

            if (
                referenced_law in EXPECTED_LAWS
                and referenced_law != assigned_law
            ):

                # Cross-law references are common.
                # Only flag if the chunk appears to contain
                # a strong main section marker.
                main_heading_pattern = re.compile(
                    rf"(?im)^\s*LAW\s+{referenced_law}\s+"
                )

                if main_heading_pattern.search(
                    text
                ):

                    problems.append(
                        {
                            "chunk_id": item["id"],
                            "assigned_law": assigned_law,
                            "possible_law": referenced_law,
                            "page": item["page"]
                        }
                    )

    return problems


# ============================================================
# BUILD CHUNKS
# ============================================================

def create_chunks():

    print("=" * 70)
    print("CRICKETSENSE-AI MCC KNOWLEDGE BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # Check PDF
    # --------------------------------------------------------

    if not os.path.exists(PDF_PATH):

        raise FileNotFoundError(
            f"\nMCC PDF not found:\n{PDF_PATH}\n"
        )

    print()
    print("Reading PDF:")
    print(PDF_PATH)

    # --------------------------------------------------------
    # Read PDF
    # --------------------------------------------------------

    reader = PdfReader(
        PDF_PATH
    )

    print(
        f"Total PDF pages: "
        f"{len(reader.pages)}"
    )

    # ========================================================
    # FIRST PASS
    # GLOBAL LAW BOUNDARIES
    # ========================================================

    print()
    print(
        "Building MCC Law boundaries..."
    )
    print()

    boundaries = build_global_law_boundaries(
        reader
    )

    detected_laws = validate_boundaries(
        boundaries
    )

    if set(detected_laws) != EXPECTED_LAWS:

        raise RuntimeError(
            "\nERROR: Could not reliably detect "
            "all 42 MCC Laws.\n"
            "Fix heading detection before creating "
            "the knowledge base."
        )

    # ========================================================
    # SECOND PASS
    # BUILD SECTIONS
    # ========================================================

    print()
    print(
        "Creating Law-specific sections..."
    )

    data = []

    chunk_id = 0

    current_law = None

    # --------------------------------------------------------
    # Process each page
    # --------------------------------------------------------

    for page_no, page in enumerate(
        reader.pages,
        start=1
    ):

        if page_no in TOC_PAGES:

            continue

        text = page.extract_text()

        if not text:

            continue

        text = clean_text(
            text
        )

        if not text:

            continue

        sections, current_law = extract_page_sections(
            text,
            page_no,
            current_law
        )

        # ----------------------------------------------------
        # Process sections
        # ----------------------------------------------------

        for section in sections:

            law_number = section[
                "law_number"
            ]

            section_text = section[
                "text"
            ].strip()

            if (
                not section_text
                or law_number is None
            ):

                continue

            # ------------------------------------------------
            # Split section into chunks
            # ------------------------------------------------

            chunks = split_large_text(
                section_text,
                CHUNK_SIZE,
                OVERLAP
            )

            for chunk in chunks:

                if not chunk.strip():

                    continue

                data.append(
                    {
                        "id": chunk_id,
                        "page": page_no,
                        "law_number": law_number,
                        "law_title": LAW_TITLES[
                            law_number
                        ],
                        "text": chunk.strip()
                    }
                )

                chunk_id += 1

    # ========================================================
    # REMOVE INVALID DATA
    # ========================================================

    cleaned_data = []

    for item in data:

        if not item["text"].strip():

            continue

        if (
            item["law_number"] not in
            EXPECTED_LAWS
        ):

            continue

        cleaned_data.append(
            item
        )

    # Re-number IDs
    for index, item in enumerate(
        cleaned_data
    ):

        item["id"] = index

    data = cleaned_data

    # ========================================================
    # SAVE JSON
    # ========================================================

    output_directory = os.path.dirname(
        OUTPUT_FILE
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    distribution = Counter(
        item["law_number"]
        for item in data
    )

    print()
    print("=" * 70)
    print("MCC CHUNK BUILD COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Total chunks : {len(data)}"
    )

    print(
        f"Output file  : {OUTPUT_FILE}"
    )

    print()
    print(
        "Law distribution:"
    )

    print()

    for law_number in range(1, 43):

        print(
            f"Law {law_number:2d}: "
            f"{distribution.get(law_number, 0)} chunks"
        )

    # ========================================================
    # FINAL LAW VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    final_laws = set(
        distribution.keys()
    )

    missing = sorted(
        EXPECTED_LAWS - final_laws
    )

    unexpected = sorted(
        final_laws - EXPECTED_LAWS
    )

    if not missing and not unexpected:

        print(
            "SUCCESS: All expected Law numbers "
            "are present."
        )

    else:

        if missing:

            print(
                "Missing Law numbers:",
                missing
            )

        if unexpected:

            print(
                "Unexpected Law numbers:",
                unexpected
            )

    # ========================================================
    # CROSS-LAW CHECK
    # ========================================================

    print()
    print(
        "Checking for cross-Law contamination..."
    )

    contamination = detect_cross_law_contamination(
        data
    )

    if not contamination:

        print(
            "SUCCESS: No obvious cross-Law "
            "heading contamination detected."
        )

    else:

        print(
            f"WARNING: {len(contamination)} "
            "possible cross-Law contamination "
            "cases detected."
        )

        for problem in contamination[:20]:

            print(
                f"  Chunk {problem['chunk_id']} | "
                f"assigned Law {problem['assigned_law']} | "
                f"possible Law {problem['possible_law']} | "
                f"page {problem['page']}"
            )

    # ========================================================
    # LARGEST LAW
    # ========================================================

    if distribution:

        largest_law = max(
            distribution,
            key=distribution.get
        )

        largest_count = distribution[
            largest_law
        ]

        print()
        print(
            f"Largest Law chunk count: "
            f"Law {largest_law} = "
            f"{largest_count} chunks"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    create_chunks()
```
