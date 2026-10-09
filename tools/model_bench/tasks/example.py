"""Template for a task module. Copy it, replace the tasks, run it with bench.py.

Each task is one prompt, sent unchanged to every model, and one scorer that turns
the answer into four pass/fail points. Prefer checks a machine can make (valid
JSON, exact values, limits kept); keep at most one point for something you have
to read yourself, and score it by hand afterwards.

Before a real run, feed every scorer the correct answer and a wrong one and make
sure it gives 4 and less than 4.
"""
from __future__ import annotations

import json
import re

PROMPTS = {
    "classify": (
        "Classify each task with one label from: CODE, LEVEL, ART, TEXT, BUG. Output only a JSON array "
        "of 6 labels, in order. 1) Spool queue skips a colour after undo. 2) Write Turkish text for the "
        "shop button. 3) Make level 12 easier. 4) Block shadow looks flat. 5) Extract drag logic into its "
        "own controller. 6) Game freezes when two linked shapes thaw together."
    ),
    "spool_math": (
        "A level has red blocks holding yarn amounts 12, 7 and 9, and blue blocks holding 5 and 14. Every "
        "spool holds exactly 10. Give the minimum number of spools needed per colour and the unused spool "
        'capacity left per colour. JSON only: {"red_spools":n,"blue_spools":n,"red_unused":n,"blue_unused":n}'
    ),
}


def parse_json(text: str):
    """Return (value, strict): strict is True when the answer was nothing but JSON."""
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
        end = max(text.rfind("}"), text.rfind("]"))
        candidate = text[min(starts):end + 1] if starts and end > min(starts) else None
    if candidate:
        try:
            return json.loads(candidate), False
        except json.JSONDecodeError:
            return None, False
    return None, False


def score_classify(text: str) -> list[bool]:
    value, strict = parse_json(text)
    want = ["BUG", "TEXT", "LEVEL", "ART", "CODE", "BUG"]
    got = [str(label).upper() for label in value] if isinstance(value, list) else []
    right = sum(1 for a, b in zip(got, want) if a == b)
    return [strict and isinstance(value, list), len(got) == 6, right >= 5, got == want]


def score_spool_math(text: str) -> list[bool]:
    value, _ = parse_json(text)
    data = value if isinstance(value, dict) else {}
    return [
        isinstance(value, dict),
        data.get("red_spools") == 3 and data.get("blue_spools") == 2,
        data.get("red_unused") == 2,
        data.get("blue_unused") == 1,
    ]


SCORERS = {"classify": score_classify, "spool_math": score_spool_math}

# Tasks that need a picture: {"task_name": "relative/or/absolute/path.png"}.
# They run on the vision-capable models only.
IMAGES: dict[str, str] = {}
