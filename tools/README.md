# Page generators

The animations and the results charts in `index.html` are generated:

- `gen_anims.py`   the three explanatory animations (targets, sample efficiency, loss band)
- `gen_flow.py`    the loss-computation animation (imports helpers from `gen_anims.py`)
- `gen_results.py` the Results charts; the numbers live at the top of this file

Each script rewrites its own block of `index.html` in place and is safe to re-run:

    python3 tools/gen_anims.py && python3 tools/gen_flow.py && python3 tools/gen_results.py

Standalone preview pages are written to `build/` (git-ignored). The scene player
behind the animations is `static/js/anim.js`; run its tests with `node --test tests/`.
