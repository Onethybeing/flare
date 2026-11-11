"""Supplementary notebook = setup/data/model/group sections of btp_main.py + extra_body.py.
Usage: python make_extra.py [quick|full]"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbtools import write_nb, write_meta

here = os.path.dirname(os.path.abspath(__file__))
quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
code = open(os.path.join(here, "..", "main", "btp_main.py"), encoding="utf-8").read()
code = code.replace("QUICK = False", f"QUICK = {quick}", 1)
parts = re.split(r"(?m)^(?=# %% \[\d+\])", code)
S = {re.match(r"# %% \[(\d+)\]", p).group(1): p for p in parts if p.startswith("# %% [")}
gnn_defs = S["6"].split("XP = {fs")[0]
tune_defs = S["10"].split("E3rows, CFGS")[0]
body = open(os.path.join(here, "extra_body.py"), encoding="utf-8").read()
extra_cells = [c for c in re.split(r"(?m)^(?=# %% \[X)", body) if c.strip()]
cells = [S["0"], S["1"], S["2"], "XVL = XV\n" + S["4"], gnn_defs, S["8"], S["9"], tune_defs] + extra_cells
cells = [c.replace("# %% ", "# ", 1) for c in cells]
src = "\n".join(cells); compile(src, "extra", "exec")
write_nb(cells, os.path.join(here, "extra.ipynb"))
slug = "btp-flowgroup-extra-quick" if quick else "btp-flowgroup-extra"
write_meta(here, slug, slug, "extra.ipynb", gpu=True)
print("ok", slug, len(cells), "cells")
