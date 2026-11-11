"""Build the Kaggle notebook from btp_main.py. Usage: python make_main.py [quick|full]"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbtools import write_nb, write_meta

here = os.path.dirname(os.path.abspath(__file__))
mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
code = open(os.path.join(here, "btp_main.py"), encoding="utf-8").read()
code = code.replace("QUICK = False", f"QUICK = {mode == 'quick'}", 1)
cells = [c for c in re.split(r"(?m)^# %% ", code) if c.strip()]
cells = ["# " + c for c in cells]
write_nb(cells, os.path.join(here, "btp_main.ipynb"))
slug = "btp-flowgroup-quick" if mode == "quick" else "btp-flowgroup-full"
write_meta(here, slug, slug, "btp_main.ipynb", gpu=True)
print("wrote", len(cells), "cells,", slug)
