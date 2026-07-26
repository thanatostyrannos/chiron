"""Cross-package integration tests: enforce the dependency-direction boundary.

This is the lint contract named in CLAUDE.md. Mnemosyne (the memory subsystem) is a
contribution only if it is separable from Proteus, and separability is enforced
mechanically rather than by discipline: an accidental `import proteus` inside
mnemosyne must fail CI, not survive review.

Filesystem-based on purpose — it needs no installed packages and no torch, so it
runs green on the CPU-only scaffold before the ROCm stack exists.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MNEMOSYNE_SRC = REPO_ROOT / "packages" / "mnemosyne" / "src"
FORBIDDEN_TOP_LEVEL = {"proteus", "themis"}


def _imported_top_level_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Ignore relative imports (node.level > 0); only absolute imports carry a module.
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def test_mnemosyne_never_imports_proteus_or_themis() -> None:
    offenders: list[str] = []
    for py_file in MNEMOSYNE_SRC.rglob("*.py"):
        imported = _imported_top_level_modules(py_file.read_text(encoding="utf-8"))
        for forbidden in sorted(imported & FORBIDDEN_TOP_LEVEL):
            offenders.append(f"{py_file.relative_to(REPO_ROOT)} imports '{forbidden}'")
    assert not offenders, (
        "Mnemosyne must depend on torch only (CLAUDE.md boundary rule). Forbidden "
        "imports found:\n  " + "\n  ".join(offenders)
    )
