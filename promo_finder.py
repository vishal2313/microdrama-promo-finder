"""Promo finder for microdrama screenplays.
Pipeline: parse (code) -> per-episode LLM candidates -> code validation/snapping -> global LLM ranking (+ cross-episode stitches) -> outputs.
Swap the model/provider with env vars: PROVIDER=gemini|openai, MODEL=<name>. Keys: GEMINI_API_KEY / OPENAI_API_KEY.
"""
import os, re, json, csv, time, sys
from parse import load_all, parse_episode  # noqa

PROVIDER = os.environ.get("PROVIDER", "gemini")
MODEL = os.environ.get("MODEL", "gemini-2.5-flash")   # change to any model your key can use
MIN_S, MAX_S = 20.0, 60.0
MAX_SEGS, MIN_SEG = 3, 4.0
PER_EP = 3          # candidates requested per episode
SLEEP = float(os.environ.get("SLEEP", "4"))  # seconds between calls (free-tier rate limits)

# ---------- LLM layer (the only provider-specific code) ----------
def llm_json(prompt, retries=4):
    for i in range(retries):
        try:
            txt = _call(prompt)
            txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
            return json.loads(txt)
        except Exception as e:
            print(f"  LLM/JSON error ({e}); retry {i+1}/{retries}", file=sys.stderr)
            time.sleep(SLEEP * (i + 2))
    return None

def as_dict(r, key):
    """Models sometimes return a bare list instead of {key: [...]}; accept both."""
    if isinstance(r, list): return {key: r}
    return r if isinstance(r, dict) else {}

def flatten_candidates(r):
    """Unwrap odd shapes: {'candidates':[{'candidates':[...]}]} or lists of lists."""
    out = []
    def walk(x):
        if isinstance(x, list):
            for y in x: walk(y)
        elif isinstance(x, dict):
            if "segments" in x: out.append(x)
            elif "candidates" in x: walk(x["candidates"])
    walk(r)
    return out

def _call(prompt):
    if PROVIDER == "gemini":
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        r = client.models.generate_content(model=MODEL, contents=prompt,
                                           config={"response_mime_type": "application/json", "temperature": 0.3})
        return r.text
    if PROVIDER == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(model=MODEL, temperature=0.3,
            response_format={"type": "json_object"}, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content
    raise ValueError("unknown PROVIDER")

# ---------- transcript helpers ----------
def mmss(t): return f"{int(t//60):02d}:{t%60:04.1f}"
def flat_items(ep):
    out = []
    for s in ep["scenes"]:
        for it in s["items"]:
            if it["type"] in ("DIALOGUE", "ACTION", "SOUND FX"):
                out.append(dict(it, scene_end=s["end"]))
    return sorted(out, key=lambda x: x["start"])

def transcript(ep):
    lines = []
    for s in ep["scenes"]:
        lines.append(f"[SCENE {s['heading']} {mmss(s['start'])}-{mmss(s['end'])}]")
        for it in s["items"]:
            if it["type"] == "CAMERA": continue
            tag = it["who"] + (f" ({it['tone']})" if it["tone"] else "") if it["type"] == "DIALOGUE" else it["type"]
            lines.append(f"{mmss(it['start'])}-{mmss(it['end'])} {tag}: {' '.join(it['text'].split())}")
    return "\n".join(lines)

CRITERIA = f"""A PROMO is a {MIN_S:.0f}-{MAX_S:.0f} second clip (total across segments) posted on TikTok/Instagram/YouTube to pull in NEW viewers.
It must: (1) open on a HOOK in the first 3 seconds (slap, reveal, shocking line, public humiliation) - not a quiet establishing shot;
(2) END on an unresolved CLIFFHANGER (cut right before the answer/reaction), (3) make sense to someone who has never seen the show;
use only timestamps that exist in the transcript. Segments must be in story order. STRONGLY prefer ONE continuous stretch (a single segment). Use at most 2 segments, and only when the join still reads naturally; never use more than 3. Every segment must be at least 4 seconds long. Do not skip over the key line or reveal just to reach the ending."""

def cand_prompt(ep):
    return f"""You are a short-drama promo editor.
{CRITERIA}

EPISODE {ep['ep']} LOGLINE: {ep['logline']}
CONTINUITY: {ep['continuity']}

TRANSCRIPT (timestamps mm:ss.s):
{transcript(ep)}

Return a JSON object with up to {PER_EP} DIFFERENT promo candidates (fewer if the episode has no good one; never pad). Different candidates should use different moments of the episode. JSON:
{{"candidates":[{{"segments":[{{"start":"mm:ss.s","end":"mm:ss.s"}}],"hook":"what happens in first 3s","cliffhanger":"what is left unresolved at the cut","why":"one line","score":0-10,"needs_context":"none|low|high"}}]}}"""

# ---------- validation (code, not AI) ----------
def p(t):
    """Accepts mm:ss.s, hh:mm:ss.s, or plain seconds."""
    if isinstance(t, (int, float)): return float(t)
    parts = str(t).strip().split(":")
    v = 0.0
    for x in parts: v = v * 60 + float(x)
    return v

def snap_segments(ep, segs):
    items = flat_items(ep)
    if not items: return None, "no items"
    starts = [i["start"] for i in items]; ends = [i["end"] for i in items]
    total = ep["scenes"][-1]["end"]; out = []
    for sg in segs:
        a, b = p(sg["start"]), p(sg["end"])
        a = min(starts, key=lambda x: abs(x - a)); b = min(ends, key=lambda x: abs(x - b))
        if b <= a: return None, "end<=start"
        out.append((a, min(b, total)))
    for (a1, b1), (a2, b2) in zip(out, out[1:]):
        if a2 < b1: return None, "overlapping/out-of-order segments"
    return out, None

def text_in(ep, a, b, first=True, window=3.0):
    its = flat_items(ep)
    sel = [i for i in its if (a <= i["start"] < a + window)] if first else [i for i in its if (b - window <= i["end"] <= b + 0.01)]
    return " | ".join(f"{i['who'] or i['type']}: {' '.join(i['text'].split())}" for i in sel)[:300]

def build(ep, c, pid):
    segs, err = snap_segments(ep, c.get("segments", []))
    if err: return None, err
    if len(segs) > MAX_SEGS: return None, f"{len(segs)} segments (max {MAX_SEGS})"
    if any(b - a < MIN_SEG for a, b in segs): return None, f"segment shorter than {MIN_SEG:.0f}s"
    dur = sum(b - a for a, b in segs)
    if not (MIN_S <= dur <= MAX_S): return None, f"duration {dur:.1f}s outside {MIN_S:.0f}-{MAX_S:.0f}"
    last_end = segs[-1][1]
    items = flat_items(ep)
    ends_scene = any(abs(last_end - s["end"]) < 0.6 for s in ep["scenes"])
    return dict(id=pid, episode=ep["ep"], segments=[dict(start=mmss(a), end=mmss(b)) for a, b in segs],
                duration_s=round(dur, 1), hook=c.get("hook", ""), cliffhanger=c.get("cliffhanger", ""),
                why=c.get("why", ""), llm_score=c.get("score"), needs_context=c.get("needs_context", ""),
                hook_text_first3s=text_in(ep, segs[0][0], segs[0][1], True),
                last_lines=text_in(ep, segs[-1][0], last_end, False, 4.0),
                ends_at_scene_or_episode_end=ends_scene), None

# ---------- stages ----------
def measured_duration(ep, c):
    segs, err = snap_segments(ep, c.get("segments", []))
    return (sum(b - a for a, b in segs), segs) if segs else (0.0, None)

def repair_prompt(ep, c, dur, err):
    segs = c.get("segments", [])
    return f"""You previously proposed this promo for episode {ep['ep']}, but it FAILED validation: {err}.
Measured length by code: {dur:.1f}s (required {MIN_S:.0f}-{MAX_S:.0f}s total). Your arithmetic was wrong; trust the code.
Proposed segments: {json.dumps(segs)}
Hook: {c.get('hook','')}
Cliffhanger: {c.get('cliffhanger','')}

Fix it: keep the SAME dramatic moment, hook and cliffhanger, but move the start earlier / end later (or trim) so the total is about 30-45s.
Do NOT cut past the cliffhanger. Prefer ONE continuous segment. Each segment at least {MIN_SEG:.0f}s. Use only timestamps in the transcript.

TRANSCRIPT:
{transcript(ep)}

JSON: {{"segments":[{{"start":"mm:ss.s","end":"mm:ss.s"}}],"hook":"..","cliffhanger":"..","why":"..","score":0-10,"needs_context":"none|low|high"}}"""

def overlaps_existing(good, ep_no, segs):
    for g in good:
        if g["episode"] != ep_no: continue
        for gs in g["segments"]:
            ga, gb = p(gs["start"]), p(gs["end"])
            for a, b in segs:
                if min(b, gb) - max(a, ga) > 0.5 * min(b - a, gb - ga): return True
    return False

def stage_repair(eps_by, good, rejected, outdir):
    fixable = [r for r in rejected if str(r.get("reason", "")).startswith(("duration", "segment shorter")) and "candidate" in r]
    print(f"Repair pass: {len(fixable)} candidates")
    still = [r for r in rejected if r not in fixable]
    for n, r in enumerate(fixable):
        ep = eps_by[r["episode"]]; c = r["candidate"]
        dur, _ = measured_duration(ep, c)
        cp = f"{outdir}/cache/{MODEL.replace('/', '_')}_repair_ep{ep['ep']:03d}_{n:03d}.json"
        if os.path.exists(cp): out = json.load(open(cp))
        else:
            out = llm_json(repair_prompt(ep, c, dur, r["reason"])); time.sleep(SLEEP)
            if out:
                os.makedirs(os.path.dirname(cp), exist_ok=True); json.dump(out, open(cp, "w"))
        cands = flatten_candidates(out) if out else []
        if not cands: still.append(dict(r, repair="LLM failed")); continue
        item, err = build(ep, cands[0], f"E{ep['ep']:03d}-R{n+1}")
        if not item: still.append(dict(r, repair=err)); continue
        segs = [(p(x["start"]), p(x["end"])) for x in item["segments"]]
        if overlaps_existing(good, ep["ep"], segs): still.append(dict(r, repair="duplicate of existing promo")); continue
        item["repaired"] = True; item["original_problem"] = r["reason"]; good.append(item)
    return good, still

def stage_candidates(eps, outdir):
    good, rejected = [], []
    for ep in eps:
        print(f"Episode {ep['ep']}...")
        cp = f"{outdir}/cache/{MODEL.replace('/', '_')}_ep{ep['ep']:03d}.json"
        if os.path.exists(cp):
            r = json.load(open(cp)); print("  (cached)")
        else:
            r = as_dict(llm_json(cand_prompt(ep)), "candidates"); time.sleep(SLEEP)
            if r:
                os.makedirs(os.path.dirname(cp), exist_ok=True); json.dump(r, open(cp, "w"))
        if not r: rejected.append(dict(episode=ep["ep"], reason="LLM failed")); continue
        for k, c in enumerate(flatten_candidates(r)):
            try: item, err = build(ep, c, f"E{ep['ep']:03d}-{k+1}")
            except Exception as e: item, err = None, f"bad candidate format: {e}"
            (good.append(item) if item else rejected.append(dict(episode=ep["ep"], candidate=c, reason=err)))
    return good, rejected

def tail_lines(ep, a, b, span=15.0):
    return [i for i in flat_items(ep) if i["end"] > b - span and i["start"] < b]

def stage_trim(cands, by, outdir):
    """Independent end-checker: sees only the last ~15s of each clip (not the model's own claims)."""
    cp = f"{outdir}/cache/{MODEL.replace('/', '_')}_trim.json"
    if os.path.exists(cp): res = json.load(open(cp)); print("Trim check (cached)")
    else:
        res = {}
        for k in range(0, len(cands), 25):
            chunk = cands[k:k + 25]; brief = []
            for c in chunk:
                ep = by[c["episode"]]; last = c["segments"][-1]; a, b = p(last["start"]), p(last["end"])
                lines = [f"{mmss(i['start'])}-{mmss(i['end'])} {(i['who'] or i['type'])}: {' '.join(i['text'].split())[:140]}"
                         for i in tail_lines(ep, a, b) if i["end"] <= b + 0.05]
                brief.append(dict(id=c["id"], ending=lines))
            prompt = f"""You check the ENDINGS of short-drama promo clips. A good promo ends on an UNRESOLVED beat: a question just asked, a threat just made, a reveal about to land, an action mid-way. It is bad if the final lines already give the answer, carry out the threat, or deliver a resolved/mic-drop line.
For each clip below, read only its ending lines. Decide:
- "keep": the ending is already unresolved.
- "trim": the answer/resolution appears inside the ending; propose cut_after = the END timestamp (mm:ss.s, copied exactly from a line shown) of the last line to KEEP, so the clip stops just BEFORE the resolution. Choose the line that leaves the question/threat hanging.
Be conservative: trim only when clearly resolved. Never invent timestamps.
Endings: {json.dumps(brief)}
JSON: {{"results":[{{"id":"..","verdict":"keep|trim","cut_after":"mm:ss.s or null","reason":".."}}]}}"""
            out = llm_json(prompt); time.sleep(SLEEP)
            rows = out.get("results", []) if isinstance(out, dict) else (out or [])
            for r in rows:
                if isinstance(r, dict) and "id" in r: res[r["id"]] = r
        os.makedirs(os.path.dirname(cp), exist_ok=True); json.dump(res, open(cp, "w"))
    n = 0
    for c in cands:
        r = res.get(c["id"])
        if not r or r.get("verdict") != "trim" or not r.get("cut_after"): continue
        try:
            ep = by[c["episode"]]; segs = [(p(x["start"]), p(x["end"])) for x in c["segments"]]
            a, b = segs[-1]; cut = p(r["cut_after"])
            ends = [i["end"] for i in tail_lines(ep, a, b)]
            cut = min(ends, key=lambda x: abs(x - cut))
            if cut <= a + MIN_SEG or cut >= b - 0.5: continue
            new = segs[:-1] + [(a, cut)]
            if sum(y - x for x, y in new) < MIN_S: continue
            c["original_end"] = mmss(b); c["trimmed"] = True; c["trim_reason"] = r.get("reason", "")
            c["segments"] = [dict(start=mmss(x), end=mmss(y)) for x, y in new]
            c["duration_s"] = round(sum(y - x for x, y in new), 1)
            c["last_lines"] = text_in(ep, a, cut, False, 4.0); n += 1
        except Exception as e: print("  trim skipped", c["id"], e)
    print(f"Trim check: {n} of {len(cands)} endings trimmed")
    return cands

def stage_rank(cands, eps):
    by = {e["ep"]: e for e in eps}
    brief = [dict(id=c["id"], ep=c["episode"], dur=c["duration_s"], hook=c["hook"], cliff=c["cliffhanger"],
                  score=c["llm_score"], ctx=c["needs_context"], repaired=bool(c.get("repaired")), trimmed=bool(c.get("trimmed")), first3s=c["hook_text_first3s"], last=c["last_lines"]) for c in cands]
    loglines = {e["ep"]: e["logline"] for e in eps}
    prompt = f"""You are head of marketing for a short-drama app. Below are promo candidates from different episodes of ONE show.
{CRITERIA}
IMPORTANT: judge each clip by its ACTUAL opening text (first3s) and ACTUAL ending text (last), NOT by the hook/cliff descriptions, which can be stale or flattering. For repaired=true candidates the description may describe a different moment than the clip really opens on: rank them down if first3s is quiet, expository, or needs backstory, or if last is a resolved line or a mic-drop rather than an unresolved moment.
Rank ALL candidates best to worst for posting FIRST (first impressions for cold viewers: strongest hook + cleanest cliffhanger + least context needed; also prefer variety of storylines among the top 10).
Give a one-line reason each. Optionally propose up to 5 STITCHED promos joining moments from 2-3 candidates' episodes only if the join makes sense to a new viewer; reference candidate ids and say how they join.
Episode loglines: {json.dumps(loglines)}
Candidates: {json.dumps(brief)}
JSON: {{"ranking":[{{"id":"..","reason":".."}}],"stitch_ideas":[{{"ids":["..",".."],"idea":".."}}]}}"""
    r = as_dict(llm_json(prompt), "ranking")
    order = {x["id"]: (i + 1, x.get("reason", "")) for i, x in enumerate(r.get("ranking", [])) if isinstance(x, dict) and "id" in x}
    for c in cands:
        c["rank"], c["rank_reason"] = order.get(c["id"], (None, "not ranked by LLM"))
    ranked = sorted(cands, key=lambda c: (c["rank"] is None, c["rank"] or 0, -(c["llm_score"] or 0)))
    unranked = sum(1 for c in ranked if c["rank"] is None)
    for i, c in enumerate(ranked): c["rank"] = i + 1
    print(f"Ranked {len(ranked) - unranked} by LLM, {unranked} unranked (placed last)")
    return ranked, r.get("stitch_ideas", [])

def write_outputs(ranked, stitches, rejected, outdir):
    os.makedirs(outdir, exist_ok=True)
    json.dump(ranked, open(f"{outdir}/promos.json", "w"), indent=2)
    json.dump(rejected, open(f"{outdir}/rejected_candidates.json", "w"), indent=2)
    json.dump(stitches, open(f"{outdir}/stitch_ideas.json", "w"), indent=2)
    with open(f"{outdir}/promos.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rank", "id", "episode", "segments", "duration_s", "hook", "cliffhanger", "why", "rank_reason", "needs_context", "ends_at_scene_end"])
        for c in ranked:
            w.writerow([c["rank"], c["id"], c["episode"], "; ".join(f"{s['start']}-{s['end']}" for s in c["segments"]), c["duration_s"],
                        c["hook"], c["cliffhanger"], c["why"], c["rank_reason"], c["needs_context"], c["ends_at_scene_or_episode_end"]])
    with open(f"{outdir}/ranked_promos.md", "w") as f:
        f.write(f"# Ranked promos (model: {PROVIDER}/{MODEL})\n\n")
        for c in ranked:
            f.write(f"## #{c['rank']} - Episode {c['episode']} ({c['id']}, {c['duration_s']}s)\n")
            f.write("- Segments: " + ", ".join(f"{s['start']}-{s['end']}" for s in c["segments"]) + "\n")
            f.write(f"- Hook: {c['hook']}\n- Cliffhanger: {c['cliffhanger']}\n- Why it works: {c['why']}\n- Rank reason: {c['rank_reason']}\n- Context needed: {c['needs_context']}\n\n")

def run(pack, outdir="outputs", only=None):
    eps = load_all(pack)
    if only: eps = [e for e in eps if e["ep"] in only]
    cands, rejected = stage_candidates(eps, outdir)
    cands, rejected = stage_repair({e['ep']: e for e in eps}, cands, rejected, outdir)
    cands = stage_trim(cands, {e['ep']: e for e in eps}, outdir)
    ranked, stitches = stage_rank(cands, eps)
    write_outputs(ranked, stitches, rejected, outdir)
    print(f"Done: {len(ranked)} valid promos, {len(rejected)} rejected -> {outdir}/")

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "outputs",
        [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None)
