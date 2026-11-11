"""Tiny helper: build a .ipynb from a list of code-cell strings."""
import json, sys

def write_nb(cells, path):
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": c.strip("\n").splitlines(True)} for c in cells],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

def write_meta(folder, slug, title, code_file, gpu=False, datasets=("muhammadabrar78/elliptic-dataset",)):
    meta = {"id": f"onethybeing/{slug}", "title": title, "code_file": code_file, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
            "dataset_sources": list(datasets), "competition_sources": [], "kernel_sources": []}
    with open(f"{folder}/kernel-metadata.json", "w") as f:
        json.dump(meta, f, indent=1)
