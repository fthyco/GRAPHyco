"""Repository documentation layout and relative-link integrity."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_and_generated_notebook_assets_absent():
    assert not list(ROOT.rglob("*.ipynb"))
    assert not (ROOT / "notebooks").exists()
    assert not (ROOT / "notebooks/tutorials/my_model_benchmark.json").exists()


def test_usage_and_verification_directories_have_required_index():
    for name in ("docs", "usage", "verification"):
        assert (ROOT / name).is_dir()
    for name in (
        "README.md",
        "usage/README.md",
        "usage/STATIC.md",
        "usage/MONITORING.md",
        "usage/VISUALIZATION.md",
        "verification/README.md",
        "verification/CUDA.md",
        "docs/DOCUMENTATION.md",
    ):
        assert (ROOT / name).is_file(), name


def test_markdown_local_links_resolve():
    docs = [ROOT / "README.md"]
    for name in ("docs", "usage", "verification"):
        docs.extend(sorted((ROOT / name).rglob("*.md")))
    missing = []
    for document in docs:
        text = document.read_text(encoding="utf-8")
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)", text):
            target = unquote(match.group(1))
            if target.startswith(("https://", "http://", "mailto:", "#", "data:")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            if not (document.parent / target).exists():
                missing.append((str(document.relative_to(ROOT)), target))
    assert not missing, missing
