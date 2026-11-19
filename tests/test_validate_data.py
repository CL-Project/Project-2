"""Tests for the ``--output`` path handling in ``scripts/validate_data.py``."""

from scripts.validate_data import REPO_ROOT, display_path


def test_display_path_inside_repo_is_relative():
    assert display_path(REPO_ROOT / "docs" / "DATA.md") == "docs/DATA.md"


def test_display_path_outside_repo_is_absolute(tmp_path):
    target = tmp_path / "data.md"
    assert display_path(target) == str(target)
