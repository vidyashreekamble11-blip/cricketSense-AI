"""
CricketSense-AI
General Hypothetical Cricket Reasoning Engine

Purpose:
    Prepare a cricket scenario for legal reasoning using retrieved
    MCC Laws evidence.

IMPORTANT:
    This module does NOT invent MCC Law text.
    It does NOT make a final umpiring decision merely from keywords.

    It performs:
        1. Scenario event extraction
        2. Concept extraction
        3. Legal issue identification
        4. Candidate Law identification
        5. Retrieved Law analysis
        6. Law interaction planning
        7. Questions for the final legal reasoner
        8. Evidence-based decision guidance

The final decision must be based on the retrieved MCC Law text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Set, Optional


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Event:
    order: int
    text: str
    concepts: List[str] = field(default_factory=list)


@dataclass
class LegalIssue:
    name: str
    description: str
    concepts: List[str] = field(default_factory=list)
    candidate_laws: List[str] = field(default_factory=list)


@dataclass
class LawAssessment:
    law_number: str
    relevance: str
    reasons: List[str] = field(default_factory=list)
    matched_concepts: List[str] = field(default_factory=list)
    evidence_found: bool = False


@dataclass
class ReasoningPlan:
    original_question: str

    events: List[Event] = field(
        default_factory=list
    )

    concepts: List[str] = field(
        default_factory=list
    )

    candidate_laws: List[str] = field(
        default_factory=list
    )

    legal_issues: List[LegalIssue] = field(
        default_factory=list
    )

    law_assessments: List[LawAssessment] = field(
        default_factory=list
    )

    interaction_groups: List[List[str]] = field(
        default_factory=list
    )

    reasoning_questions: List[str] = field(
        default_factory=list
    )

    evidence_findings: List[str] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# CONCEPT DETECTION
# ============================================================

CONCEPT_PATTERNS = {

    "delivery": [
        r"\bdeliver(?:s|ed|y|ing)?\b",
        r"\bbowler\b",
        r"\bdelivery\b",
    ],

    "fair_delivery": [
        r"\bfair ball\b",
        r"\bfair delivery\b",
        r"\blegal delivery\b",
    ],

    "no_ball": [
        r"\bno[- ]?ball\b",
        r"\billegal delivery\b",
        r"\bfront foot\b",
        r"\bwaist[- ]high\b",
        r"\bover[- ]head\b",
    ],

    "wide": [
        r"\bwide ball\b",
        r"\bwide\b",
    ],

    "ball": [
        r"\bball\b",
    ],

    "ball_changes": [
        r"\bsplit\b",
        r"\bsplits\b",
        r"\bsplitting\b",
        r"\bbroke\b",
        r"\bbroken\b",
        r"\bbreaks\b",
        r"\bpieces\b",
        r"\btwo pieces\b",
        r"\bthree pieces\b",
        r"\bdisintegrat\w*\b",
        r"\bfractur\w*\b",
        r"\bchanges shape\b",
        r"\bchanged shape\b",
        r"\bchanges into two\b",
        r"\bdivides into two\b",
    ],

    "bat_contact": [
        r"\bhit(?:s)? the ball\b",
        r"\bstruck the ball\b",
        r"\bstrikes the ball\b",
        r"\bwith the bat\b",
        r"\bcontact with the bat\b",
        r"\bbat contact\b",
    ],

    "catch": [
        r"\bcaught\b",
        r"\bcatch\b",
        r"\bcatches\b",
        r"\bcatching\b",
        r"\bcatch the piece\b",
        r"\bcatch the pieces\b",
    ],

    "fielder": [
        r"\bfielder\b",
        r"\bfielders\b",
    ],

    "wicketkeeper": [
        r"\bwicket[- ]?keeper\b",
        r"\bkeeper\b",
    ],

    "wicket": [
        r"\bwicket\b",
        r"\bstumps?\b",
        r"\bbails?\b",
    ],

    "bowled": [
        r"\bbowled\b",
        r"\bhit(?:s)? the stumps\b",
        r"\bhit(?:s)? the wicket\b",
    ],

    "run_out": [
        r"\brun[- ]?out\b",
        r"\brun out\b",
    ],

    "stumped": [
        r"\bstumped\b",
        r"\bstumping\b",
    ],

    "lbw": [
        r"\blbw\b",
        r"\bleg before wicket\b",
        r"\bleg[- ]before\b",
    ],

    "hit_ball_twice": [
        r"\bhit(?:s)? the ball twice\b",
        r"\bhit(?:s)? it twice\b",
        r"\bsecond hit\b",
        r"\bsecond strike\b",
    ],

    "obstruction": [
        r"\bobstruct\w*\b",
        r"\binterfer\w*\b",
    ],

    "deliberate_action": [
        r"\bdeliberately\b",
        r"\bdeliberate\b",
        r"\bwilfully\b",
        r"\bwilful\b",
        r"\bintentionally\b",
        r"\bintentional\b",
    ],

    "accidental_action": [
        r"\baccidentally\b",
        r"\baccidental\b",
        r"\bunintentionally\b",
        r"\bunintentional\b",
    ],

    "dead_ball": [
        r"\bdead ball\b",
        r"\bball becomes dead\b",
        r"\bball was dead\b",
    ],

    "boundary": [
        r"\bboundary\b",
        r"\bboundaries\b",
        r"\bfour\b",
        r"\bsix\b",
    ],

    "ground_contact": [
        r"\bhits the ground\b",
        r"\btouches the ground\b",
        r"\btouch(?:es|ed)? the ground\b",
    ],

    "appeal": [
        r"\bappeal\b",
        r"\bappealed\b",
        r"\bhow'?s that\b",
    ],

    "batter": [
        r"\bbatter\b",
        r"\bstriker\b",
        r"\bnon[- ]striker\b",
    ],

    "running": [
        r"\brun\b",
        r"\bruns\b",
        r"\brunning\b",
        r"\bcrossed\b",
        r"\bcrease\b",
    ],

    "helmet": [
        r"\bhelmet\b",
        r"\bprotective helmet\b",
    ],
}


# ============================================================
# LEGAL ISSUE DEFINITIONS
# ============================================================

LEGAL_ISSUE_RULES = {

    "delivery_validity": {
        "required_any": {
            "delivery",
            "fair_delivery",
            "no_ball",
            "wide",
        },
        "laws": {
            "17",
            "21",
            "22",
        },
        "description": (
            "Determine whether the delivery was valid and whether "
            "the delivery status is affected by any later event."
        ),
    },

    "ball_status": {
        "required_any": {
            "ball_changes",
            "dead_ball",
        },
        "laws": {
            "4",
            "20",
        },
        "description": (
            "Determine whether the unusual condition of the ball "
            "has a specific legal consequence under the MCC Laws."
        ),
    },

    "catch_validity": {
        "required_any": {
            "catch",
        },
        "laws": {
            "33",
        },
        "description": (
            "Determine whether the fielding action satisfies every "
            "condition required for a fair catch."
        ),
    },

    "fielder_action": {
        "required_any": {
            "fielder",
        },
        "laws": {
            "28",
        },
        "description": (
            "Determine whether the fielder's action is governed by "
            "any fielding restriction or condition."
        ),
    },

    "bat_contact": {
        "required_any": {
            "bat_contact",
        },
        "laws": {
            "33",
            "34",
        },
        "description": (
            "Determine the legal significance of the batter's "
            "contact with the ball."
        ),
    },

    "hit_ball_twice": {
        "required_any": {
            "hit_ball_twice",
        },
        "laws": {
            "34",
        },
        "description": (
            "Determine whether the batter committed an act governed "
            "by Law 34."
        ),
    },

    "obstruction": {
        "required_any": {
            "obstruction",
        },
        "laws": {
            "37",
        },
        "description": (
            "Determine whether the described interference satisfies "
            "the conditions of Law 37."
        ),
    },

    "bowled": {
        "required_any": {
            "bowled",
        },
        "laws": {
            "32",
        },
        "description": (
            "Determine whether the conditions for Bowled are met."
        ),
    },

    "run_out": {
        "required_any": {
            "run_out",
        },
        "laws": {
            "38",
        },
        "description": (
            "Determine whether the conditions for Run out are met."
        ),
    },

    "stumped": {
        "required_any": {
            "stumped",
        },
        "laws": {
            "39",
        },
        "description": (
            "Determine whether the conditions for Stumped are met."
        ),
    },

    "lbw": {
        "required_any": {
            "lbw",
        },
        "laws": {
            "36",
        },
        "description": (
            "Determine whether the conditions for LBW are met."
        ),
    },

    "appeal": {
        "required_any": {
            "appeal",
        },
        "laws": {
            "31",
        },
        "description": (
            "Determine whether the appeal requirement affects "
            "the umpire's ability to give the batter out."
        ),
    },
}


# ============================================================
# LAW INTERACTIONS
# ============================================================

LAW_INTERACTIONS = {

    "4": {
        "17",
        "20",
        "21",
        "28",
        "33",
    },

    "17": {
        "20",
        "21",
        "22",
    },

    "20": {
        "4",
        "17",
        "21",
        "28",
        "33",
        "34",
        "37",
        "38",
        "39",
    },

    "21": {
        "17",
        "20",
        "22",
        "28",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
    },

    "28": {
        "20",
        "33",
    },

    "33": {
        "4",
        "20",
        "21",
        "28",
        "32",
        "34",
        "37",
        "38",
        "39",
    },

    "34": {
        "21",
        "33",
        "37",
    },

    "37": {
        "21",
        "33",
        "38",
    },

    "38": {
        "20",
        "21",
        "37",
    },

    "39": {
        "20",
        "21",
        "33",
        "38",
    },
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:

    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# ============================================================
# QUESTION DETECTION
# ============================================================

QUESTION_PATTERNS = [

    r"^\s*what is\b",
    r"^\s*what was\b",
    r"^\s*who is\b",
    r"^\s*who was\b",
    r"^\s*is the\b",
    r"^\s*are the\b",
    r"^\s*was the\b",
    r"^\s*were the\b",
    r"^\s*can the\b",
    r"^\s*could the\b",
    r"^\s*does the\b",
    r"^\s*did the\b",
    r"^\s*should the\b",
    r"^\s*would the\b",
    r"^\s*will the\b",
    r"^\s*how does\b",
    r"^\s*how is\b",
]


def is_question_sentence(
    sentence: str,
) -> bool:

    text = sentence.strip()

    if not text:
        return False

    if text.endswith("?"):
        return True

    lower = text.lower()

    for pattern in QUESTION_PATTERNS:

        if re.search(
            pattern,
            lower,
        ):
            return True

    return False


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_into_sentences(
    question: str,
) -> List[str]:

    question = normalize_text(
        question
    )

    if not question:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        question,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# CONCEPT EXTRACTION
# ============================================================

def detect_concepts(
    text: str,
) -> Set[str]:

    if not text:
        return set()

    lower = text.lower()

    concepts = set()

    for concept, patterns in CONCEPT_PATTERNS.items():

        for pattern in patterns:

            if re.search(
                pattern,
                lower,
                flags=re.IGNORECASE,
            ):
                concepts.add(
                    concept
                )
                break

    return concepts


# ============================================================
# EVENT EXTRACTION
# ============================================================

def extract_events(
    question: str,
) -> List[Event]:

    sentences = split_into_sentences(
        question
    )

    events = []

    order = 1

    for sentence in sentences:

        if is_question_sentence(
            sentence
        ):
            continue

        concepts = detect_concepts(
            sentence
        )

        if not concepts:
            continue

        events.append(
            Event(
                order=order,
                text=sentence,
                concepts=sorted(
                    concepts
                ),
            )
        )

        order += 1

    return events


# ============================================================
# LEGAL ISSUE DETECTION
# ============================================================

def identify_legal_issues(
    concepts: Set[str],
) -> List[LegalIssue]:

    issues = []

    for issue_name, rule in LEGAL_ISSUE_RULES.items():

        required_any = rule[
            "required_any"
        ]

        matched = (
            concepts
            & required_any
        )

        if not matched:
            continue

        issues.append(
            LegalIssue(
                name=issue_name,
                description=rule[
                    "description"
                ],
                concepts=sorted(
                    matched
                ),
                candidate_laws=sorted(
                    rule["laws"],
                    key=lambda x: int(x),
                ),
            )
        )

    return issues


# ============================================================
# CANDIDATE LAW GENERATION
# ============================================================

def generate_candidate_laws(
    legal_issues: List[LegalIssue],
) -> Set[str]:

    laws = set()

    for issue in legal_issues:

        laws.update(
            issue.candidate_laws
        )

    return laws


# ============================================================
# LAW NUMBER NORMALIZATION
# ============================================================

def normalize_law_number(
    value: Any,
) -> Optional[str]:

    if value is None:
        return None

    text = str(
        value
    ).strip()

    match = re.search(
        r"\b(\d{1,2})\b",
        text,
    )

    if not match:
        return None

    number = int(
        match.group(1)
    )

    if not 1 <= number <= 42:
        return None

    return str(
        number
    )


# ============================================================
# RETRIEVED LAW EXTRACTION
# ============================================================

def get_retrieved_laws(
    retrieved_results: List[Dict[str, Any]],
) -> Set[str]:

    laws = set()

    for result in retrieved_results:

        law = normalize_law_number(
            result.get(
                "law_number"
            )
        )

        if law:
            laws.add(
                law
            )

    return laws


# ============================================================
# LAW TEXT EXTRACTION
# ============================================================

def get_result_text(
    result: Dict[str, Any],
) -> str:

    possible_fields = [
        "text",
        "content",
        "page_content",
        "chunk",
        "law_text",
    ]

    for field_name in possible_fields:

        value = result.get(
            field_name
        )

        if value:

            return str(
                value
            )

    return ""


# ============================================================
# LAW RELEVANCE
# ============================================================

def assess_law_relevance(
    law_number: str,
    legal_issues: List[LegalIssue],
    concepts: Set[str],
    evidence_found: bool = True,
) -> LawAssessment:

    matching_issues = [
        issue
        for issue in legal_issues
        if law_number in issue.candidate_laws
    ]

    matched_concepts = set()

    reasons = []

    for issue in matching_issues:

        matched_concepts.update(
            issue.concepts
        )

        reasons.append(
            f"Relevant to the legal issue '{issue.name}'."
        )

    primary_issue_names = {
        "delivery_validity",
        "ball_status",
        "catch_validity",
        "bowled",
        "run_out",
        "stumped",
        "lbw",
        "hit_ball_twice",
        "obstruction",
    }

    primary_matches = [
        issue
        for issue in matching_issues
        if issue.name in primary_issue_names
    ]

    if primary_matches:
        relevance = "primary"

    elif matching_issues:
        relevance = "supporting"

    else:
        relevance = "contextual"

    if not reasons:

        reasons.append(
            "The Law was retrieved but was not directly "
            "identified by the legal-issue layer."
        )

    return LawAssessment(
        law_number=law_number,
        relevance=relevance,
        reasons=reasons,
        matched_concepts=sorted(
            matched_concepts
        ),
        evidence_found=evidence_found,
    )


# ============================================================
# INTERACTION GROUPS
# ============================================================

def build_interaction_groups(
    laws: Set[str],
) -> List[List[str]]:

    groups = []

    processed = set()

    for law in sorted(
        laws,
        key=lambda x: int(x),
    ):

        related = LAW_INTERACTIONS.get(
            law,
            set(),
        )

        related = related & laws

        if not related:
            continue

        group = {
            law,
            *related,
        }

        group_tuple = tuple(
            sorted(
                group,
                key=lambda x: int(x),
            )
        )

        if group_tuple in processed:
            continue

        processed.add(
            group_tuple
        )

        groups.append(
            list(group_tuple)
        )

    return groups


# ============================================================
# REASONING QUESTIONS
# ============================================================

def build_reasoning_questions(
    concepts: Set[str],
    legal_issues: List[LegalIssue],
) -> List[str]:

    questions = []

    questions.append(
        "What happened first, second, and subsequently in the scenario?"
    )

    if "delivery_validity" in {
        issue.name
        for issue in legal_issues
    }:

        questions.append(
            "Was the delivery valid, and does any later event "
            "change the validity of the delivery?"
        )

    if "ball_status" in {
        issue.name
        for issue in legal_issues
    }:

        questions.append(
            "Does the retrieved MCC text contain a specific rule "
            "governing the ball becoming split, damaged, altered, "
            "or otherwise changing condition?"
        )

        questions.append(
            "Does the unusual change in the ball cause the ball "
            "to become dead under an expressly applicable Law?"
        )

    if "catch_validity" in {
        issue.name
        for issue in legal_issues
    }:

        questions.append(
            "Does Law 33.1 require the same ball that was delivered "
            "and struck by the batter to be subsequently held as "
            "a fair catch?"
        )

        questions.append(
            "Does the retrieved MCC text expressly state what happens "
            "when the ball changes into separate pieces before a catch?"
        )

        questions.append(
            "Can catching two separate pieces by two different fielders "
            "satisfy the definition of a fair catch under the retrieved "
            "Law 33 text?"
        )

    if "fielder_action" in {
        issue.name
        for issue in legal_issues
    }:

        questions.append(
            "Does Law 28 contain any provision that changes the "
            "result of the described catches?"
        )

    if "bat_contact" in {
        issue.name
        for issue in legal_issues
    }:

        questions.append(
            "Was the ball lawfully struck by the batter before "
            "the unusual event occurred?"
        )

    questions.append(
        "After applying only the retrieved MCC provisions to the "
        "events in chronological order, is the batter out?"
    )

    return questions


# ============================================================
# EVIDENCE ANALYSIS
# ============================================================

def analyze_retrieved_evidence(
    retrieved_results: List[Dict[str, Any]],
    concepts: Set[str],
) -> List[str]:

    findings = []

    law33_texts = []
    law20_texts = []
    law17_texts = []
    law4_texts = []

    for result in retrieved_results:

        law = normalize_law_number(
            result.get(
                "law_number"
            )
        )

        text = get_result_text(
            result
        )

        if law == "33":
            law33_texts.append(
                text
            )

        elif law == "20":
            law20_texts.append(
                text
            )

        elif law == "17":
            law17_texts.append(
                text
            )

        elif law == "4":
            law4_texts.append(
                text
            )

    # --------------------------------------------------------
    # Law 33 evidence
    # --------------------------------------------------------

    combined33 = " ".join(
        law33_texts
    ).lower()

    if "33.1" in combined33:

        findings.append(
            "Retrieved Law 33.1 states that the striker is out "
            "Caught when a ball delivered by the bowler, not a No ball, "
            "touches the bat and is subsequently held by a fielder "
            "as a fair catch before it touches the ground."
        )

    if "33.2.1" in combined33:

        findings.append(
            "Retrieved Law 33.2.1 states that a catch is fair only "
            "if the specified grounding conditions are satisfied."
        )

    if "33.3" in combined33:

        findings.append(
            "Retrieved Law 33.3 defines the act of making a catch "
            "from first contact with a fielder until complete control "
            "over the ball and the fielder's own movement."
        )

    # --------------------------------------------------------
    # Important limitation
    # --------------------------------------------------------

    if (
        "split" not in combined33
        and "piece" not in combined33
        and "two pieces" not in combined33
    ):

        findings.append(
            "The retrieved Law 33 evidence does not expressly "
            "address a cricket ball splitting into two separate pieces."
        )

    # --------------------------------------------------------
    # Law 20 evidence
    # --------------------------------------------------------

    combined20 = " ".join(
        law20_texts
    ).lower()

    if combined20:

        if "becomes dead" in combined20:

            findings.append(
                "Law 20 evidence was retrieved, but the supplied "
                "excerpt does not expressly identify ball splitting "
                "as an automatic Dead ball event."
            )

    # --------------------------------------------------------
    # Law 17 evidence
    # --------------------------------------------------------

    combined17 = " ".join(
        law17_texts
    ).lower()

    if combined17:

        findings.append(
            "Law 17 concerns the over and validity/counting of balls; "
            "the supplied evidence should not be used by itself to "
            "decide whether catching split pieces constitutes a catch."
        )

    # --------------------------------------------------------
    # Law 4 evidence
    # --------------------------------------------------------

    combined4 = " ".join(
        law4_texts
    ).lower()

    if not combined4 and "ball_changes" in concepts:

        findings.append(
            "Law 4 was identified as a candidate Law, but no Law 4 "
            "text was present in the supplied retrieved evidence."
        )

    return findings


# ============================================================
# WARNINGS
# ============================================================

def build_warnings(
    events: List[Event],
    concepts: Set[str],
    legal_issues: List[LegalIssue],
    retrieved_results: List[Dict[str, Any]],
) -> List[str]:

    warnings = []

    if len(events) >= 3:

        warnings.append(
            "This is a multi-event hypothetical. "
            "The Laws must be applied chronologically."
        )

    if "ball_changes" in concepts:

        warnings.append(
            "The scenario contains an abnormal change to the ball. "
            "Do not assume that the ball automatically becomes dead "
            "unless the retrieved MCC text supports that conclusion."
        )

    if (
        "ball_changes" in concepts
        and "catch" in concepts
    ):

        warnings.append(
            "The scenario asks whether two separate pieces can constitute "
            "a catch. Law 33 must be examined carefully because the "
            "retrieved excerpt does not expressly address a split ball."
        )

    if len(legal_issues) >= 3:

        warnings.append(
            "Multiple legal issues are present. "
            "Do not answer from a single Law in isolation."
        )

    retrieved_laws = get_retrieved_laws(
        retrieved_results
    )

    if (
        "33" in candidate_laws_for_concepts(concepts)
        and "33" not in retrieved_laws
    ):

        warnings.append(
            "Law 33 is a required candidate Law for the catch scenario "
            "but was not retrieved."
        )

    return warnings


# ============================================================
# HELPER
# ============================================================

def candidate_laws_for_concepts(
    concepts: Set[str],
) -> Set[str]:

    issues = identify_legal_issues(
        concepts
    )

    return generate_candidate_laws(
        issues
    )


# ============================================================
# MAIN ANALYZER
# ============================================================

def analyze_scenario(
    question: str,
    retrieved_results: Optional[
        List[Dict[str, Any]]
    ] = None,
) -> ReasoningPlan:

    question = normalize_text(
        question
    )

    if not question:

        return ReasoningPlan(
            original_question="",
            warnings=[
                "No scenario was provided."
            ],
        )

    retrieved_results = (
        retrieved_results
        or []
    )

    # --------------------------------------------------------
    # 1. Extract events
    # --------------------------------------------------------

    events = extract_events(
        question
    )

    # --------------------------------------------------------
    # 2. Extract concepts ONLY from events
    # --------------------------------------------------------

    concepts = set()

    for event in events:

        concepts.update(
            event.concepts
        )

    # --------------------------------------------------------
    # 3. Legal issues
    # --------------------------------------------------------

    legal_issues = identify_legal_issues(
        concepts
    )

    # --------------------------------------------------------
    # 4. Candidate Laws
    # --------------------------------------------------------

    candidate_laws = generate_candidate_laws(
        legal_issues
    )

    # --------------------------------------------------------
    # 5. Retrieved Laws
    # --------------------------------------------------------

    retrieved_laws = get_retrieved_laws(
        retrieved_results
    )

    # --------------------------------------------------------
    # 6. Law assessments
    # --------------------------------------------------------

    law_assessments = []

    for law in sorted(
        candidate_laws,
        key=lambda x: int(x),
    ):

        if law not in retrieved_laws:
            continue

        assessment = assess_law_relevance(
            law_number=law,
            legal_issues=legal_issues,
            concepts=concepts,
            evidence_found=True,
        )

        law_assessments.append(
            assessment
        )

    # --------------------------------------------------------
    # 7. Evidence analysis
    # --------------------------------------------------------

    evidence_findings = (
        analyze_retrieved_evidence(
            retrieved_results,
            concepts,
        )
        if retrieved_results
        else []
    )

    # --------------------------------------------------------
    # 8. Interaction groups
    # --------------------------------------------------------

    if retrieved_results:

        interaction_laws = (
            candidate_laws
            & retrieved_laws
        )

    else:

        interaction_laws = candidate_laws

    interaction_groups = build_interaction_groups(
        interaction_laws
    )

    # --------------------------------------------------------
    # 9. Questions for final reasoner
    # --------------------------------------------------------

    reasoning_questions = build_reasoning_questions(
        concepts,
        legal_issues,
    )

    # --------------------------------------------------------
    # 10. Warnings
    # --------------------------------------------------------

    warnings = build_warnings(
        events,
        concepts,
        legal_issues,
        retrieved_results,
    )

    # --------------------------------------------------------
    # 11. Return reasoning plan
    # --------------------------------------------------------

    return ReasoningPlan(
        original_question=question,
        events=events,
        concepts=sorted(
            concepts
        ),
        candidate_laws=sorted(
            candidate_laws,
            key=lambda x: int(x),
        ),
        legal_issues=legal_issues,
        law_assessments=law_assessments,
        interaction_groups=interaction_groups,
        reasoning_questions=reasoning_questions,
        evidence_findings=evidence_findings,
        warnings=warnings,
    )


# ============================================================
# HUMAN-READABLE OUTPUT
# ============================================================

def format_reasoning_plan(
    plan: ReasoningPlan,
) -> str:

    lines = []

    lines.append(
        "=" * 70
    )

    lines.append(
        "CRICKETSENSE-AI REASONING PLAN"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"\nQUESTION\n{plan.original_question}"
    )

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    lines.append(
        "\nEVENT SEQUENCE"
    )

    lines.append(
        "-" * 70
    )

    if plan.events:

        for event in plan.events:

            concepts = (
                ", ".join(
                    event.concepts
                )
                if event.concepts
                else "none"
            )

            lines.append(
                f"{event.order}. {event.text}"
            )

            lines.append(
                f"   Concepts: {concepts}"
            )

    else:

        lines.append(
            "No cricket events detected."
        )

    # --------------------------------------------------------
    # Concepts
    # --------------------------------------------------------

    lines.append(
        "\nSCENARIO CONCEPTS"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        ", ".join(
            plan.concepts
        )
        if plan.concepts
        else "None"
    )

    # --------------------------------------------------------
    # Legal Issues
    # --------------------------------------------------------

    lines.append(
        "\nLEGAL ISSUES"
    )

    lines.append(
        "-" * 70
    )

    if plan.legal_issues:

        for issue in plan.legal_issues:

            laws = ", ".join(
                f"Law {law}"
                for law in issue.candidate_laws
            )

            lines.append(
                f"\n• {issue.name}"
            )

            lines.append(
                f"  {issue.description}"
            )

            lines.append(
                "  Concepts: "
                + ", ".join(
                    issue.concepts
                )
            )

            lines.append(
                f"  Candidate Laws: {laws}"
            )

    else:

        lines.append(
            "No legal issues identified."
        )

    # --------------------------------------------------------
    # Candidate Laws
    # --------------------------------------------------------

    lines.append(
        "\nCANDIDATE LAWS"
    )

    lines.append(
        "-" * 70
    )

    if plan.candidate_laws:

        lines.append(
            ", ".join(
                f"Law {law}"
                for law in plan.candidate_laws
            )
        )

    else:

        lines.append(
            "None"
        )

    # --------------------------------------------------------
    # Retrieved Law Relevance
    # --------------------------------------------------------

    lines.append(
        "\nRETRIEVED LAW RELEVANCE"
    )

    lines.append(
        "-" * 70
    )

    if plan.law_assessments:

        for assessment in plan.law_assessments:

            lines.append(
                f"Law {assessment.law_number}: "
                f"{assessment.relevance.upper()}"
            )

            if assessment.matched_concepts:

                lines.append(
                    "  Matched concepts: "
                    + ", ".join(
                        assessment.matched_concepts
                    )
                )

            for reason in assessment.reasons:

                lines.append(
                    f"  - {reason}"
                )

    else:

        lines.append(
            "No candidate Laws were found in retrieved evidence."
        )

    # --------------------------------------------------------
    # Evidence Findings
    # --------------------------------------------------------

    lines.append(
        "\nRETRIEVED MCC EVIDENCE ANALYSIS"
    )

    lines.append(
        "-" * 70
    )

    if plan.evidence_findings:

        for finding in plan.evidence_findings:

            lines.append(
                f"- {finding}"
            )

    else:

        lines.append(
            "No retrieved evidence was supplied."
        )

    # --------------------------------------------------------
    # Interaction Groups
    # --------------------------------------------------------

    lines.append(
        "\nLAW INTERACTION GROUPS"
    )

    lines.append(
        "-" * 70
    )

    if plan.interaction_groups:

        for group in plan.interaction_groups:

            lines.append(
                " + ".join(
                    f"Law {law}"
                    for law in group
                )
            )

    else:

        lines.append(
            "No interaction groups detected."
        )

    # --------------------------------------------------------
    # Final Reasoner Questions
    # --------------------------------------------------------

    lines.append(
        "\nQUESTIONS FOR FINAL LEGAL REASONER"
    )

    lines.append(
        "-" * 70
    )

    for index, question in enumerate(
        plan.reasoning_questions,
        start=1,
    ):

        lines.append(
            f"{index}. {question}"
        )

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    lines.append(
        "\nWARNINGS"
    )

    lines.append(
        "-" * 70
    )

    if plan.warnings:

        for warning in plan.warnings:

            lines.append(
                f"- {warning}"
            )

    else:

        lines.append(
            "None"
        )

    lines.append(
        "\n" + "=" * 70
    )

    return "\n".join(
        lines
    )


# ============================================================
# FINAL REASONING INPUT BUILDER
#
# This function prepares a clean prompt/context for the LLM.
# It does NOT itself decide OUT/NOT OUT.
# ============================================================

def build_final_reasoner_context(
    plan: ReasoningPlan,
    retrieved_results: List[Dict[str, Any]],
) -> str:

    lines = []

    lines.append(
        "CRICKETSENSE-AI FINAL LEGAL REASONING CONTEXT"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        "\nSCENARIO:"
    )

    lines.append(
        plan.original_question
    )

    lines.append(
        "\nEVENTS IN CHRONOLOGICAL ORDER:"
    )

    for event in plan.events:

        lines.append(
            f"{event.order}. {event.text}"
        )

    lines.append(
        "\nLEGAL ISSUES:"
    )

    for issue in plan.legal_issues:

        lines.append(
            f"- {issue.name}: {issue.description}"
        )

        lines.append(
            "  Candidate Laws: "
            + ", ".join(
                f"Law {law}"
                for law in issue.candidate_laws
            )
        )

    lines.append(
        "\nRETRIEVED MCC LAW EVIDENCE:"
    )

    if retrieved_results:

        for index, result in enumerate(
            retrieved_results,
            start=1,
        ):

            law = result.get(
                "law_number",
                "Unknown",
            )

            page = result.get(
                "page",
                "Unknown",
            )

            score = result.get(
                "score",
                "Unknown",
            )

            text = get_result_text(
                result
            )

            lines.append(
                f"\n[{index}] Law {law} | "
                f"Page {page} | Score {score}"
            )

            lines.append(
                text
            )

    else:

        lines.append(
            "No MCC evidence supplied."
        )

    lines.append(
        "\nFINAL REASONING INSTRUCTIONS:"
    )

    lines.append(
        "1. Use only the retrieved MCC evidence for legal conclusions."
    )

    lines.append(
        "2. Do not invent a rule for an event that the retrieved "
        "MCC text does not address."
    )

    lines.append(
        "3. Apply the Laws chronologically."
    )

    lines.append(
        "4. Identify the exact condition required for the proposed "
        "dismissal."
    )

    lines.append(
        "5. For a Caught decision, examine Law 33.1, 33.2 and 33.3."
    )

    lines.append(
        "6. Specifically determine whether the retrieved evidence "
        "supports treating two separate pieces caught by two fielders "
        "as the ball being held as a fair catch."
    )

    lines.append(
        "7. If the retrieved evidence is insufficient to answer the "
        "split-ball issue, explicitly say that the evidence is "
        "insufficient rather than inventing a conclusion."
    )

    lines.append(
        "8. Give the final decision only after explaining the "
        "applicable Law and its conditions."
    )

    return "\n".join(
        lines
    )


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "CricketSense-AI Reasoning Engine"
    )

    print(
        "Type a cricket scenario and press Enter."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        try:

            question = input(
                "\nScenario: "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):

            print(
                "\nExiting."
            )

            break

        if question.lower() in {
            "exit",
            "quit",
        }:

            break

        plan = analyze_scenario(
            question
        )

        print(
            format_reasoning_plan(
                plan
            )
        )

        print(
            "\nFINAL REASONER CONTEXT"
        )

        print(
            "=" * 70
        )

        print(
            build_final_reasoner_context(
                plan,
                [],
            )
        )