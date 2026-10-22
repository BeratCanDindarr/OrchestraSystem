"""Video teardown benchmark: can a model read motion and timing from a sheet of video frames?

Each task is one contact sheet of 8 frames (30 fps, 33 ms apart), drawn by make_sheets() with a
known answer, standing in for frames ffmpeg extracts from a gameplay video. Ollama vision models
get the sheet as an image (IMAGES); the CLI agents read the same file from CLI_CWD, see
video_frames_cli.py.

    python tools/model_bench/tasks/video_frames.py      # draw the sheets and self-check the scorers
"""
from __future__ import annotations

import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SHEET_DIR = os.path.join(HERE, "video_frames")
FRAME_W, FRAME_H, COLS = 320, 320, 4
FRAME_MS = 33

SHUFFLE_SEED = 7
SHUFFLED = list(range(8))
random.Random(SHUFFLE_SEED).shuffle(SHUFFLED)  # SHUFFLED[i] = true time index shown in cell i
LETTERS = "ABCDEFGH"
TRUE_ORDER = [LETTERS[SHUFFLED.index(t)] for t in range(8)]

SHEET_NOTE = ("The image is a contact sheet of 8 consecutive frames from a mobile-game screen recording, "
              "read left to right, top row first; each frame's label shows its number and time (30 fps). ")

PROMPTS = {
    "bounce": SHEET_NOTE + (
        "A ball falls onto the floor and bounces. Report the frame where it touches the floor, the frame "
        "where it is stretched tallest, the milliseconds from F1 to the floor contact, and whether it moves "
        'up or down after contact. JSON only: {"impact_frame":n,"stretch_frame":n,"impact_ms":n,"moving_after":"up|down"}'
    ),
    "punch": SHEET_NOTE + (
        "A button plays a punch-scale animation while a coin counter above it counts up. Report the frame "
        "where the button is smallest, the frame where it is largest, the first frame where the counter "
        'value is higher than in F1, and the counter value in F8. JSON only: '
        '{"smallest_frame":n,"largest_frame":n,"counter_start_frame":n,"counter_final":n}'
    ),
    "shake": SHEET_NOTE + (
        "At some point the screen shakes (the whole scene is offset from its resting place) and a white "
        "flash burst appears. Report the first and last shaken frame and the frame with the flash burst. "
        'JSON only: {"shake_first":n,"shake_last":n,"flash_frame":n}'
    ),
    "order": (
        "The image is a contact sheet of 8 frames from a mobile-game screen recording, labelled A to H, "
        "but shuffled: the labels are NOT in time order. A ball of yarn flies along an arc from the left "
        "to a spool on the right, shrinking as it goes, while a progress bar fills. Put the labels in time "
        'order, earliest first. JSON only: {"order":["A",...]}'
    ),
}
IMAGES = {name: os.path.join(SHEET_DIR, f"{name}.png") for name in PROMPTS}


def parse_json(text: str):
    """Return (value, strict): strict is True when the answer was nothing but JSON."""
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start:end + 1] if 0 <= start < end else None
    if candidate:
        try:
            return json.loads(candidate), False
        except json.JSONDecodeError:
            return None, False
    return None, False


def _data(text: str) -> dict:
    value, _ = parse_json(text)
    return value if isinstance(value, dict) else {}


def score_bounce(text: str) -> list[bool]:
    data = _data(text)
    ms = data.get("impact_ms")
    return [
        data.get("impact_frame") == 4,
        data.get("stretch_frame") == 5,
        isinstance(ms, (int, float)) and 90 <= ms <= 110,
        str(data.get("moving_after", "")).lower() == "up",
    ]


def score_punch(text: str) -> list[bool]:
    data = _data(text)
    return [
        data.get("smallest_frame") == 3,
        data.get("largest_frame") == 5,
        data.get("counter_start_frame") == 4,
        data.get("counter_final") == 125,
    ]


def score_shake(text: str) -> list[bool]:
    data = _data(text)
    return [
        data.get("shake_first") == 4,
        data.get("shake_last") == 6,
        data.get("flash_frame") == 4,
        bool(data) and parse_json(text)[1],
    ]


def score_order(text: str) -> list[bool]:
    order = _data(text).get("order")
    got = [str(x).upper() for x in order] if isinstance(order, list) else []
    right = sum(1 for a, b in zip(got, TRUE_ORDER) if a == b)
    return [
        sorted(got) == list(LETTERS),
        got[:1] == TRUE_ORDER[:1] and got[-1:] == TRUE_ORDER[-1:],
        right >= 6,
        got == TRUE_ORDER,
    ]


SCORERS = {"bounce": score_bounce, "punch": score_punch, "shake": score_shake, "order": score_order}

RIGHT_ANSWERS = {
    "bounce": '{"impact_frame":4,"stretch_frame":5,"impact_ms":99,"moving_after":"up"}',
    "punch": '{"smallest_frame":3,"largest_frame":5,"counter_start_frame":4,"counter_final":125}',
    "shake": '{"shake_first":4,"shake_last":6,"flash_frame":4}',
    "order": json.dumps({"order": TRUE_ORDER}),
}
WRONG_ANSWERS = {
    "bounce": '{"impact_frame":5,"stretch_frame":4,"impact_ms":132,"moving_after":"up"}',
    "punch": '{"smallest_frame":2,"largest_frame":4,"counter_start_frame":4,"counter_final":125}',
    "shake": 'Sure: {"shake_first":3,"shake_last":6,"flash_frame":4}',
    "order": json.dumps({"order": list(LETTERS)}),
}


def make_sheets() -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=26)
    small = ImageFont.load_default(size=22)
    os.makedirs(SHEET_DIR, exist_ok=True)

    def sheet(frames, labels, name):
        rows = (len(frames) + COLS - 1) // COLS
        out = Image.new("RGB", (COLS * FRAME_W + (COLS + 1) * 8, rows * (FRAME_H + 36) + 8), (40, 40, 40))
        for i, (frame, label) in enumerate(zip(frames, labels)):
            x, y = 8 + (i % COLS) * (FRAME_W + 8), 8 + (i // COLS) * (FRAME_H + 36)
            out.paste(frame, (x, y + 30))
            ImageDraw.Draw(out).text((x + 4, y), label, fill=(255, 255, 255), font=font)
        out.save(os.path.join(SHEET_DIR, f"{name}.png"))

    timed = [f"F{i + 1}  {i * FRAME_MS} ms" for i in range(8)]

    def blank():
        return Image.new("RGB", (FRAME_W, FRAME_H), (120, 170, 220))

    # bounce: floor at y=280; contact (squashed) in F4, stretched in F5, rising after
    ball = [(60, 40, 40), (110, 40, 40), (170, 40, 40), (255, 58, 25), (215, 30, 52), (160, 40, 40),
            (120, 40, 40), (95, 40, 40)]  # (centre y, half width, half height)
    frames = []
    for cy, hw, hh in ball:
        image = blank()
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 280, FRAME_W, FRAME_H), fill=(90, 60, 40))
        cy = min(cy, 280 - hh)
        draw.ellipse((160 - hw, cy - hh, 160 + hw, cy + hh), fill=(220, 50, 60))
        frames.append(image)
    sheet(frames, timed, "bounce")

    # punch: button scale per frame, counter above
    scales = [1.00, 0.90, 0.84, 1.14, 1.22, 1.08, 0.98, 1.00]
    counts = [120, 120, 120, 121, 122, 123, 124, 125]
    frames = []
    for scale, count in zip(scales, counts):
        image = blank()
        draw = ImageDraw.Draw(image)
        draw.text((110, 30), f"coins {count}", fill=(255, 230, 0), font=font)
        hw, hh = 90 * scale, 45 * scale
        draw.rounded_rectangle((160 - hw, 190 - hh, 160 + hw, 190 + hh), radius=18 * scale, fill=(60, 190, 80))
        draw.text((160 - 30 * scale, 178), "PLAY", fill=(255, 255, 255), font=small)
        frames.append(image)
    sheet(frames, timed, "punch")

    # shake: scene offset in F4-F6, flash in F4
    offsets = [(0, 0), (0, 0), (0, 0), (14, -10), (-12, 9), (8, -6), (0, 0), (0, 0)]
    frames = []
    for i, (dx, dy) in enumerate(offsets):
        image = blank()
        draw = ImageDraw.Draw(image)
        draw.rectangle((40 + dx, 200 + dy, 280 + dx, 300 + dy), fill=(150, 100, 60))
        for k in range(3):
            draw.rectangle((70 + k * 70 + dx, 150 + dy, 110 + k * 70 + dx, 200 + dy), fill=(200, 80 + 50 * k, 90))
        draw.line((0 + dx, 300 + dy, FRAME_W + dx, 300 + dy), fill=(30, 30, 30), width=4)
        if i == 3:
            for a in range(12):
                draw.line((160 + dx, 120 + dy, 160 + dx + (a % 4 - 1.5) * 50, 120 + dy + (a // 4 - 1) * 60),
                          fill=(255, 255, 255), width=6)
            draw.ellipse((130 + dx, 90 + dy, 190 + dx, 150 + dy), fill=(255, 255, 255))
        frames.append(image)
    sheet(frames, timed, "shake")

    # order: arc flight, shrinking, progress bar; cells shuffled
    timeline = []
    for t in range(8):
        image = blank()
        draw = ImageDraw.Draw(image)
        draw.rectangle((250, 200, 300, 270), fill=(170, 120, 70))  # spool
        u = t / 7
        x, y, r = 40 + 220 * u, 220 - 160 * 4 * u * (1 - u), 34 - 22 * u
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(230, 90, 160))
        draw.rectangle((20, 295, 300, 310), outline=(255, 255, 255), width=2)
        draw.rectangle((22, 297, 22 + 276 * u, 308), fill=(255, 255, 255))
        timeline.append(image)
    sheet([timeline[t] for t in SHUFFLED], list(LETTERS), "order")


def self_check() -> None:
    for name, scorer in SCORERS.items():
        good, bad = sum(scorer(RIGHT_ANSWERS[name])), sum(scorer(WRONG_ANSWERS[name]))
        assert good == 4 and bad < 4, (name, good, bad)
    print("scorers ok; true order:", "".join(TRUE_ORDER))


if __name__ == "__main__":
    make_sheets()
    self_check()
