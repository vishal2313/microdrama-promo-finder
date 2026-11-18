import re, glob, os, json
PACK = "/home/claude/pack/I Married a Broken Soldier. He was a Duke"
def ts(s):
    m, r = s.split(":"); return int(m)*60 + float(r)
def parse_episode(path):
    t = open(path, encoding="utf-8").read()
    ep = int(re.search(r"episode-(\d+)", os.path.basename(path)).group(1))
    head = t.split("--- SCRIPT START ---")[0]
    logline = re.search(r"EPISODE LOGLINE:\s*(.*)", head)
    cont = re.search(r"CONTINUITY NOTES:\s*(.*)", head)
    body = t.split("--- SCRIPT START ---")[1].split("--- SCRIPT END ---")[0]
    scenes = []
    for blk in re.split(r"\n(?=SCENE \d+)", body.strip(), flags=re.I):
        h = re.search(r"Scene heading:\s*(.*)", blk, re.I)
        if not h: continue
        s = re.search(r"Scene start:\s*([\d:.]+)", blk, re.I); e = re.search(r"Scene end:\s*([\d:.]+)", blk, re.I)
        items = []
        for m in re.finditer(r"\*\*(.+?):\*\*\s*\nStart:\s*\(([\d:.]+)\)\s*\|\s*End:\s*\(([\d:.]+)\)\s*\n(.*?)(?=\n\*\*|\Z)", blk, re.S):
            label, a, b, txt = m.groups()
            kind = label.split(" (")[0].strip()
            typ = kind if kind in ("CAMERA","ACTION","SOUND FX","MUSIC","SFX") else "DIALOGUE"
            items.append(dict(type=typ, who=kind if typ=="DIALOGUE" else "", tone=(re.search(r"\((.*)\)", label).group(1) if "(" in label else ""),
                              start=ts(a), end=ts(b), text=txt.strip()))
        scenes.append(dict(heading=h.group(1).strip(), start=ts(s.group(1)), end=ts(e.group(1)), items=items))
    return dict(ep=ep, logline=logline.group(1) if logline else "", continuity=cont.group(1) if cont else "", scenes=scenes)
def load_all(pack=PACK):
    return [parse_episode(p) for p in sorted(glob.glob(os.path.join(pack, "episode-*.txt")))]
if __name__ == "__main__":
    eps = load_all()
    for e in eps[:50]:
        dur = e["scenes"][-1]["end"]; n = sum(len(s["items"]) for s in e["scenes"])
        print(e["ep"], len(e["scenes"]), "scenes", n, "items", f"{dur:.0f}s")
