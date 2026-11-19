"""Scripts that import ``src`` must run as documented: ``python scripts/<name>.py`` (AGENTS.md §5).

Running a file puts its own directory on ``sys.path``, not the repository root, so each such script
bootstraps the root itself.  ``--help`` proves the import graph loads and exits without touching data.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_SCRIPTS = sorted(
    path for path in (ROOT / "scripts").glob("*.py") if "from src" in path.read_text(encoding="utf-8")
)


@pytest.mark.parametrize("script", SRC_SCRIPTS, ids=lambda path: path.name)
def test_help_runs_from_repo_root(script):
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
