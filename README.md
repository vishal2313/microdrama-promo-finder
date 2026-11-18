# Microdrama Promo Finder

Reads timestamped microdrama screenplays and finds 20-60 s promo clips (hook in the first 3 s, unresolved cliffhanger, makes sense to a new viewer). Built for the Kuku AI Generalist assignment, Problem 1.

## How it works
1. `parse.py` parses the screenplays into scenes and timestamped lines (code).
2. An LLM proposes up to 3 candidates per episode.
3. Code snaps times to real lines and enforces 20-60 s, at most 3 segments, at least 4 s per segment.
4. A repair pass re-sends clips that failed on length, with the measured duration.
5. An end check (LLM sees only the last ~15 s) trims endings that already give the answer.
6. An LLM ranks by the real opening and ending text.

Per-episode answers are cached, so an interrupted run resumes.

## Run it (Google Colab)
1. Open `promo_finder_colab.ipynb` in Colab.
2. Add a Colab Secret named `GEMINI_API_KEY` (free key from aistudio.google.com) and enable notebook access.
3. Upload `parse.py`, `promo_finder.py` and the screenplay `.zip` to the Colab files panel.
4. Run the cells in order. Set `MODEL` to a model your key can use (the notebook lists them).
5. Outputs appear in `outputs/`: `ranked_promos.md`, `promos.csv`, `promos.json`, `rejected_candidates.json`, `stitch_ideas.json`.

## Swap the model
Set `PROVIDER` (`gemini` or `openai`) and `MODEL` in the setup cell. The OpenAI path is written but untested.

## Limits
Judges from text only (no video). Cross-episode stitches are suggested, not built. Some episodes still get no valid promo. The LLM end check can miss resolved endings. See the solution doc for the quality results and failures.

The screenplays are Kuku's content and are not included in this repo.
