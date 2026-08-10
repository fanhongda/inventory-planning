"""
Every text file this package reads or writes is UTF-8, stated explicitly.

On Linux and macOS the default locale is already UTF-8, so an unencoded `read_text()`
works and nothing shows up in CI. On Windows the default is the ANSI code page —
cp1252 for a Western install — and the same call raises. `config/planning_parameters.md`
is written in Chinese, so the failure is immediate and total: the parameter file cannot
be read, and `suggested_rules_<ts>.md` cannot be written.

That is a real production break with no local symptom, which is exactly the kind of
bug a source-level test has to catch. `PYTHONUTF8=1` papers over it but leaves the
package dependent on how the caller happened to launch Python.
"""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "inventory_planning"
CONFIG = ROOT / "config"

# Calls whose default encoding is locale-dependent.
TEXT_IO = {"read_text", "write_text"}


def _python_files():
    return sorted(PACKAGE.rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))


def _unencoded_calls(path: Path):
    """(line, call) for every text read/write that does not state its encoding."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        keywords = {k.arg for k in node.keywords}
        if "encoding" in keywords:
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in TEXT_IO:
            out.append((node.lineno, func.attr))
        # bare open(...) in text mode
        elif isinstance(func, ast.Name) and func.id == "open":
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            if "b" not in mode:
                out.append((node.lineno, "open"))
        elif isinstance(func, ast.Attribute) and func.attr == "open":
            mode = ""
            if node.args and isinstance(node.args[0], ast.Constant):
                mode = str(node.args[0].value)
            if "b" not in mode:
                out.append((node.lineno, "Path.open"))
    return out


class TestNoLocaleDependentTextIO:

    def test_no_unencoded_text_io_anywhere(self):
        offenders = []
        for path in _python_files():
            for line, call in _unencoded_calls(path):
                offenders.append(f"{path.relative_to(ROOT)}:{line} {call}()")
        assert not offenders, (
            "these calls fall back to the platform encoding and break on Windows:\n  "
            + "\n  ".join(offenders)
        )


class TestConfigFilesAreUtf8:

    @pytest.mark.parametrize("name", ["planning_parameters.md", "stocking_policy.json",
                                      "incoterm_rules.json", "node_config.json",
                                      "supplier_incoterm.json"])
    def test_config_decodes_as_utf8(self, name):
        (CONFIG / name).read_text(encoding="utf-8")

    def test_planning_parameters_really_is_non_ascii(self):
        """
        If this file ever became pure ASCII the encoding tests above would pass
        vacuously and the guard would rot.
        """
        text = (CONFIG / "planning_parameters.md").read_text(encoding="utf-8")
        assert not text.isascii(), "the Windows failure this guards depends on non-ASCII"


class TestNonAsciiRoundTrip:
    """The specific operations that failed in production, exercised directly."""

    def test_parameter_file_parses(self):
        from inventory_planning.policy.parameters import PlanningParameters
        assert PlanningParameters(CONFIG / "planning_parameters.md").rules

    def test_rule_heading_with_a_middle_dot_is_readable(self, tmp_path):
        """`R-001 · no reason` is byte 0xb7 — undecodable under cp1252."""
        from inventory_planning.policy.parameters import PlanningParameters
        path = tmp_path / "p.md"
        path.write_text(
            "## Conventions\n```yaml\ncycle_stock_basis: peak\n"
            "safety_stock_exposure: review_plus_lt\npipeline_basis: none\n```\n"
            "## Rules\n### R-001 · no reason\n```yaml\nscope: abc_class == \"A\"\n"
            "set:\n  review_period_days: 7\n```\n",
            encoding="utf-8",
        )
        # Must fail on the missing rationale — the business rule — not on decoding.
        with pytest.raises(ValueError, match="rationale"):
            PlanningParameters(path)

    def test_suggested_rules_markdown_writes_non_ascii(self, tmp_path):
        """`to_rules_markdown` emits a Chinese heading and an arrow."""
        from inventory_planning.policy.suggestions import SuggestionResult
        result = SuggestionResult(frame=_minimal_frame(), rules=[], hits={})
        path = tmp_path / "suggested_rules.md"
        result.to_rules_markdown(path)
        assert "建议规则" in path.read_text(encoding="utf-8")


def _minimal_frame():
    import pandas as pd
    return pd.DataFrame({"sku": ["A-1"], "lt_samples": [3], "changes": [""]})
