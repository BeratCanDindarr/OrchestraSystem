"""Benchmark of Ollama cloud models on non-review tasks. Three requests run at a time."""
import base64, json, os, re, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

OUT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(OUT, "raw")
os.makedirs(RAW, exist_ok=True)
CAP = 75  # seconds per call
SHOT = os.path.join(os.path.dirname(OUT), "link.png")

# model -> think level sent (None = not sent)
MODELS = {
    "glm-5.3:cloud": "low",
    "glm-5.3-flash:cloud": "low",
    "deepseek-v4-pro:cloud": False,
    "deepseek-v4.1-flash:cloud": False,
    "minimax-m3:cloud": False,
    "minimax-m2.7:cloud": None,
    "mistral-large-4:cloud": "low",
    "gemma4:31b-cloud": False,
    "nemotron-3-nano:30b-cloud": False,
    "nemotron-3-super:cloud": False,
    "nemotron-3-ultra:cloud": False,
    "gpt-oss:120b-cloud": "low",
    "gpt-oss:20b-cloud": "low",
}
VISION = ["glm-5.3-flash:cloud", "deepseek-v4.1-flash:cloud", "minimax-m3:cloud", "mistral-large-4:cloud", "gemma4:31b-cloud"]

PROMPTS = {
    "level": 'Design one level for a puzzle on a 6x6 grid. Output ONLY JSON: {"shapes":[{"id":1,"color":"red","amount":3,"cells":[[x,y]]}],"spools":[{"edge":"top","queue":[{"color":"red","capacity":3}]}]}. Rules: 4-6 shapes; x,y in 0..5; 2-4 cells per shape; no overlapping cells; each shape\'s cells orthogonally connected; per colour, sum of amount equals sum of capacity; edge is top, bottom, left or right.',
    "mechanics": "Game: drag multi-cell coloured blocks on a 6x6 grid; yarn pours from a block into matching-colour spools queued at the board edges. Existing extras: ice, axis locks, linked shapes. Propose exactly 5 new mechanics. For each: name, one-sentence rule, one new field needed in level data, one way it could make a level unsolvable. Max 200 words total.",
    "juice": "Unity with DOTween. A dragged block is released, snaps into its grid cell, then yarn starts pouring into a spool. Give exactly 6 juice ideas, one per line, format: moment | effect | tweened property | duration seconds | ease. Use only real DOTween Ease enum names. Each duration at most 0.5. No effects needing new art assets. No other text.",
    "plan": 'Unity 2022, C#, Zenject, DOTween. A pure C# board simulation (no Unity APIs) emits events; MonoBehaviour views render them. Plan adding an "ice" extra: a frozen shape cannot be dragged until N moves have been made. Give at most 8 numbered steps, each starting with the layer it touches in brackets: [Core], [Controller], [View], [Settings], [Editor] or [Tests]. No code.',
    "turkish": 'Translate these mobile game UI strings to Turkish. Informal "sen" tone, max 24 characters each, keep placeholders unchanged. Output only JSON with the same keys. {"level_complete":"Level Complete!","no_moves":"No moves left","retry":"Try again","reward":"You earned {0} coins","next":"Next level","hint":"Drag the block to the spool"}',
    "summary": 'Summarise these playtest notes in exactly 3 bullets, max 12 words each, most severe first. Add nothing that is not in the notes. Notes: "Level 7: two testers stuck 4 minutes, did not notice the axis lock icon. Level 3 fine. Yarn pour sound too loud for one tester. Level 9 crashed once when a linked shape was dropped on ice. Everyone liked the spool fill animation."',
    "classify": "Classify each task with one label from: CODE, LEVEL, ART, TEXT, BUG. Output only a JSON array of 6 labels, in order. 1) Spool queue skips a colour after undo. 2) Write Turkish text for the shop button. 3) Make level 12 easier. 4) Block shadow looks flat. 5) Extract drag logic into its own controller. 6) Game freezes when two linked shapes thaw together.",
    "vision": 'Look at this game screenshot. Answer with JSON only: {"grid_cols":n,"grid_rows":n,"shape_count":n,"colors":["..."],"ui_problems":["up to 3 short strings"]}. Count only what is visible (shapes are the coloured blocks lying ON the board, not the spools around it). Use null where you are not sure.',
}

EASES = {"Unset","Linear","InSine","OutSine","InOutSine","InQuad","OutQuad","InOutQuad","InCubic","OutCubic","InOutCubic","InQuart","OutQuart","InOutQuart","InQuint","OutQuint","InOutQuint","InExpo","OutExpo","InOutExpo","InCirc","OutCirc","InOutCirc","InElastic","OutElastic","InOutElastic","InBack","OutBack","InOutBack","InBounce","OutBounce","InOutBounce","Flash","InFlash","OutFlash","InOutFlash"}


def call(model, think, prompt, image=None):
    body = {"model": model, "prompt": prompt, "stream": True}
    if think is not None:
        body["think"] = think
    if image:
        body["images"] = [image]
    start = time.time(); text = []; thinking = 0; first = None; err = ""; out_tok = None
    try:
        req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                d = json.loads(line)
                thinking += len(d.get("thinking") or "")
                t = d.get("response") or ""
                if t:
                    first = first or (time.time() - start)
                    text.append(t)
                if d.get("error"):
                    err = str(d["error"])[:200]; break
                if d.get("done"):
                    out_tok = d.get("eval_count"); break
                if time.time() - start > CAP:
                    err = "cap"; break
    except urllib.error.HTTPError as e:
        err = "HTTP %s %s" % (e.code, e.read().decode(errors="replace")[:160])
    except Exception as e:
        err = str(e)[:160]
    return {"seconds": round(time.time() - start, 1), "first": first and round(first, 1), "thinking_chars": thinking, "out_tok": out_tok, "error": err, "text": "".join(text).strip()}


def parse_json(text):
    """(value, strict). strict = the answer was nothing but JSON."""
    try:
        return json.loads(text), True
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    cand = m.group(1) if m else None
    if cand is None:
        a = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
        b = max(text.rfind("}"), text.rfind("]"))
        cand = text[a:b + 1] if a >= 0 and b > a else None
    if cand:
        try:
            return json.loads(cand), False
        except Exception:
            return None, False
    return None, False


def connected(cells):
    cells = {tuple(c) for c in cells}
    if not cells: return False
    seen = set(); todo = [next(iter(cells))]
    while todo:
        c = todo.pop()
        if c in seen: continue
        seen.add(c)
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            n = (c[0]+dx, c[1]+dy)
            if n in cells: todo.append(n)
    return seen == cells


def score_level(t):
    v, strict = parse_json(t); pts = [strict and v is not None, False, False, False]
    try:
        shapes = v["shapes"]; spools = v["spools"]
        cells = [tuple(c) for s in shapes for c in s["cells"]]
        pts[1] = all(0 <= x <= 5 and 0 <= y <= 5 for x, y in cells) and len(cells) == len(set(cells))
        pts[2] = 4 <= len(shapes) <= 6 and all(2 <= len(s["cells"]) <= 4 and connected(s["cells"]) for s in shapes)
        amount = {}; cap = {}
        for s in shapes: amount[s["color"]] = amount.get(s["color"], 0) + s["amount"]
        for sp in spools:
            if sp["edge"] not in ("top","bottom","left","right"): raise ValueError
            for q in sp["queue"]: cap[q["color"]] = cap.get(q["color"], 0) + q["capacity"]
        pts[3] = amount == cap and len(amount) > 0
    except Exception:
        pass
    return pts


def score_classify(t):
    v, strict = parse_json(t)
    want = ["BUG","TEXT","LEVEL","ART","CODE","BUG"]
    ok = isinstance(v, list) and [str(x).upper() for x in v] == want
    right = sum(1 for a, b in zip(v, want) if str(a).upper() == b) if isinstance(v, list) else 0
    return [strict and isinstance(v, list), ok, right >= 5, isinstance(v, list) and len(v) == 6]


def score_turkish(t):
    v, strict = parse_json(t)
    keys = {"level_complete","no_moves","retry","reward","next","hint"}
    pts = [False]*4
    if isinstance(v, dict):
        pts[0] = strict and set(v) == keys
        vals = [str(x) for x in v.values()]
        pts[1] = set(v) == keys and "{0}" in str(v.get("reward","")) and all(len(x) <= 24 for x in vals)
        joined = " ".join(vals)
        pts[2] = any(ch in joined for ch in "çğışöüÇĞİŞÖÜ")
        pts[3] = not re.search(r"\b(siz|iniz|ınız|unuz|ünüz)\b", joined.lower()) and not re.search(r"(ınız|iniz|unuz|ünüz)\b", joined.lower())
    return pts


def score_summary(t):
    bullets = [re.sub(r"^[\-\*•\d\.\)\s]+", "", l).strip() for l in t.splitlines() if re.match(r"^\s*([\-\*•]|\d+[\.\)])\s+", l)]
    pts = [len(bullets) == 3 and all(len(b.split()) <= 12 for b in bullets), False, False, False]
    if len(bullets) >= 2:
        pts[1] = "9" in bullets[0] or "crash" in bullets[0].lower()
        pts[2] = "7" in bullets[1] or "axis" in bullets[1].lower()
    digits = set(re.findall(r"\d+", " ".join(bullets)))
    pts[3] = len(bullets) > 0 and digits <= {"3","4","7","9"}
    return pts


def score_juice(t):
    lines = [l.strip().strip("`") for l in t.splitlines() if "|" in l and not set(l.strip()) <= set("|-: ")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
    rows = [r for r in rows if r and r[0].lower() != "moment"]
    pts = [len(rows) == 6 and all(len(r) == 5 for r in rows), False, False, False]
    five = [r for r in rows if len(r) == 5]
    if five:
        pts[1] = all(r[4].replace("Ease.", "").strip() in EASES for r in five)
        def dur(x):
            m = re.search(r"\d*\.?\d+", x); return float(m.group()) if m else 99
        pts[2] = all(dur(r[3]) <= 0.5 for r in five)
        pts[3] = len({(r[0].lower(), r[2].lower()) for r in five}) >= 4
    return pts


def score_plan(t):
    steps = re.findall(r"^\s*\d+[\.\)]\s*(.*)$", t, re.M)
    tags = ("[Core]","[Controller]","[View]","[Settings]","[Editor]","[Tests]")
    clean = [re.sub(r"^[\*\s`]+", "", s) for s in steps]
    pts = [0 < len(clean) <= 8 and all(c.startswith(tags) for c in clean) and "```" not in t, False, False, False]
    core = [c for c in clean if c.startswith("[Core]")]
    pts[1] = any(re.search(r"move|count|frozen|thaw|drag", c, re.I) for c in core)
    pts[2] = any(c.startswith("[Tests]") for c in clean) and any(c.startswith("[Editor]") for c in clean)
    pts[3] = len(core) > 0 and not any(re.search(r"MonoBehaviour|Transform|GameObject|Time\.|DOTween|Zenject", c) for c in core)
    return pts


def score_mechanics(t):
    items = re.findall(r"^\s*(?:\d+[\.\)]|[\-\*•]|#+)\s*\S", t, re.M)
    numbered = re.findall(r"^\s*\**\s*\d+[\.\)]", t, re.M)
    words = len(t.split())
    low = t.lower()
    return [len(numbered) == 5 or len(items) == 5, words <= 200, "unsolvable" in low or "softlock" in low or "impossible" in low or "deadlock" in low, "field" in low or re.search(r"`\w+`|\b\w+:\s*(int|bool|float|string)", t) is not None]


def score_vision(t):
    v, strict = parse_json(t); pts = [False]*4
    if isinstance(v, dict):
        pts[0] = strict
        pts[1] = v.get("grid_cols") == 6 and v.get("grid_rows") == 6
        pts[2] = v.get("shape_count") == 3
        cols = {str(c).lower() for c in (v.get("colors") or [])}
        norm = {"orange" if c in ("yellow","orange","gold","amber") else ("red" if c in ("red","pink","coral","salmon") else c) for c in cols}
        pts[3] = norm == {"red","green","orange"}
    return pts


SCORERS = {"level": score_level, "mechanics": score_mechanics, "juice": score_juice, "plan": score_plan, "turkish": score_turkish, "summary": score_summary, "classify": score_classify, "vision": score_vision}


def job(model, task):
    image = None
    if task == "vision":
        image = base64.b64encode(open(SHOT, "rb").read()).decode()
    r = call(model, MODELS[model], PROMPTS[task], image)
    try:
        pts = [bool(x) for x in SCORERS[task](r["text"])] if r["text"] else [False]*4
    except Exception as e:
        pts = [False]*4; r["error"] += " scorer:" + str(e)[:80]
    r.update(model=model, task=task, points=pts, score=sum(pts))
    open(os.path.join(RAW, f"{task}__{model.replace(':','_')}.txt"), "w").write(r["text"])
    print(f"{task:9} {model:28} {r['score']}/4 {r['seconds']:5.1f}s think={r['thinking_chars']:6} {r['error']}", flush=True)
    return r


if __name__ == "__main__":
    jobs = [(m, t) for t in PROMPTS if t != "vision" for m in MODELS] + [(m, "vision") for m in VISION]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda a: job(*a), jobs))
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), ensure_ascii=False, indent=1)
    print("DONE", len(results), flush=True)
