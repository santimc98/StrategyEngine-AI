"""
Tests for dialect code autopatcher.

These tests verify the AST-based autopatcher for pd.read_csv dialect parameters.
No network/LLM dependencies.
"""

import os

import pytest
from src.utils.dialect_code_patch import patch_read_csv_dialect, has_kwargs_in_read_csv


os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")
os.environ.setdefault("GOOGLE_API_KEY", "dummy")
os.environ.setdefault("SANDBOX_PROVIDER", "local")
os.environ.setdefault("OPENROUTER_API_KEY", "dummy-openrouter")


class TestPatchReadCsvDialect:
    """Tests for patch_read_csv_dialect function."""

    def test_adds_missing_params(self):
        """Test that missing sep/decimal/encoding are added."""
        code = "import pandas as pd\ndf = pd.read_csv('data/raw.csv')"

        patched, notes, changed = patch_read_csv_dialect(
            code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8"
        )

        assert changed is True
        assert "sep" in patched
        assert "decimal" in patched
        assert "encoding" in patched
        assert len(notes) == 3  # Added 3 params

    def test_adds_only_missing_params(self):
        """Test that only missing params are added when some exist."""
        code = "import pandas as pd\ndf = pd.read_csv('data/raw.csv', sep=',')"

        patched, notes, changed = patch_read_csv_dialect(
            code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8"
        )

        assert changed is True
        assert "decimal" in patched
        assert "encoding" in patched
        # sep was already correct, should be 2 additions
        assert len(notes) == 2

    def test_replaces_incorrect_literal(self):
        """Test that incorrect literal values are replaced."""
        code = "import pandas as pd\ndf = pd.read_csv('data/raw.csv', sep=';')"

        patched, notes, changed = patch_read_csv_dialect(
            code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8"
        )

        assert changed is True
        assert "sep=','" in patched or "sep=\",\"" in patched
        assert any("Replaced sep" in note for note in notes)

    def test_no_change_when_correct(self):
        """Test that no changes are made when params are already correct."""
        code = "import pandas as pd\ndf = pd.read_csv('data/raw.csv', sep=',', decimal='.', encoding='utf-8')"

        patched, notes, changed = patch_read_csv_dialect(
            code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8"
        )

        assert changed is False

    def test_sanitizes_kwargs_and_pins_explicit_dialect(self):
        """Test that **kwargs is sanitized and explicit dialect is pinned."""
        code = "import pandas as pd\ndialect = {'sep': ','}\ndf = pd.read_csv('data/raw.csv', **dialect)"

        patched, notes, changed = patch_read_csv_dialect(
            code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8"
        )

        assert changed is True
        assert "_strip_csv_dialect_kwargs" in patched
        assert "sep=','" in patched or 'sep=","' in patched
        assert "decimal='.'" in patched or 'decimal="."' in patched
        assert "encoding='utf-8'" in patched or 'encoding="utf-8"' in patched
        assert any("Sanitized **kwargs" in note for note in notes)

    def test_replaces_incorrect_literal_with_kwargs(self):
        """Test that incorrect literals are replaced even with **kwargs present."""
        code = "import pandas as pd\ndialect = {}\ndf = pd.read_csv('data/raw.csv', sep=';', **dialect)"

        patched, notes, changed = patch_read_csv_dialect(
            code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8"
        )

        # sep literal is wrong, should be replaced
        assert changed is True
        assert any("Pinned sep" in note or "Replaced sep" in note for note in notes)

    def test_handles_parse_error(self):
        """Test that parse errors are handled gracefully."""
        code = "this is not valid python {{{{"

        patched, notes, changed = patch_read_csv_dialect(
            code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8"
        )

        assert changed is False
        assert patched == code
        assert any("parse error" in note.lower() for note in notes)

    def test_no_read_csv_calls(self):
        """Test handling when no read_csv calls are present."""
        code = "import pandas as pd\ndf = pd.DataFrame()"

        patched, notes, changed = patch_read_csv_dialect(
            code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8"
        )

        assert changed is False
        assert any("No pd.read_csv" in note for note in notes)

    def test_encoding_equivalence(self):
        """Test that encoding equivalences (utf-8/utf8/utf-8-sig) are respected."""
        code = "import pandas as pd\ndf = pd.read_csv('data/raw.csv', encoding='utf8')"

        patched, notes, changed = patch_read_csv_dialect(
            code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8"
        )

        # utf8 is equivalent to utf-8, so encoding should not be replaced
        # but sep and decimal are missing
        assert changed is True
        # Should NOT have replaced encoding
        assert not any("Replaced encoding" in note for note in notes)

    def test_targets_expected_path(self):
        """Test that the correct read_csv call is targeted based on path."""
        code = """
import pandas as pd
df1 = pd.read_csv('other.csv')
df2 = pd.read_csv('data/raw.csv')
"""

        patched, notes, changed = patch_read_csv_dialect(
            code,
            csv_sep=";",
            csv_decimal=",",
            csv_encoding="latin-1",
            expected_path="data/raw.csv",
        )

        assert changed is True
        # The patch should target the data/raw.csv call
        # We can verify by checking that notes mention the parameters
        assert len(notes) == 3

    def test_handles_non_literal_values(self):
        """Test that non-literal (variable) values are not touched."""
        code = """
import pandas as pd
my_sep = ';'
df = pd.read_csv('data/raw.csv', sep=my_sep)
"""

        patched, notes, changed = patch_read_csv_dialect(
            code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8"
        )

        # sep is a variable, should not be touched
        # decimal and encoding are missing, should be added
        assert changed is True
        assert not any("sep" in note.lower() and "replaced" in note.lower() for note in notes)

    def test_semicolon_separator(self):
        """Test patching with semicolon separator."""
        code = "import pandas as pd\ndf = pd.read_csv('data/raw.csv', sep=',')"

        patched, notes, changed = patch_read_csv_dialect(
            code, csv_sep=";", csv_decimal=",", csv_encoding="utf-8"
        )

        assert changed is True
        assert "sep=';'" in patched or 'sep=";"' in patched
        assert any("Replaced sep" in note for note in notes)

    def test_normalizes_manifest_dialect_alias_lookups(self):
        """Test that helper functions reading manifest dialect support sep/delimiter aliases."""
        code = """
import pandas as pd
def read_csv_with_dialect(path, dialect):
    sep = dialect.get('delimiter', ',')
    return pd.read_csv(path, sep=sep, decimal=',', encoding='utf-8')
df = read_csv_with_dialect('data/raw.csv', {'sep': ';'})
"""

        patched, notes, changed = patch_read_csv_dialect(
            code, csv_sep=";", csv_decimal=",", csv_encoding="utf-8"
        )

        assert changed is True
        assert "dialect.get('sep') or dialect.get('delimiter', ',')" in patched or 'dialect.get("sep") or dialect.get("delimiter", ",")' in patched
        assert any("sep/delimiter alias" in note for note in notes)


class TestHasKwargsInReadCsv:
    """Tests for has_kwargs_in_read_csv helper."""

    def test_detects_kwargs(self):
        """Test detection of **kwargs in read_csv."""
        code = "import pandas as pd\ndf = pd.read_csv('data/raw.csv', **dialect)"
        assert has_kwargs_in_read_csv(code) is True

    def test_no_kwargs(self):
        """Test when no **kwargs is present."""
        code = "import pandas as pd\ndf = pd.read_csv('data/raw.csv', sep=',')"
        assert has_kwargs_in_read_csv(code) is False

    def test_handles_parse_error(self):
        """Test graceful handling of parse errors."""
        code = "invalid python {{"
        assert has_kwargs_in_read_csv(code) is False


class TestDialectGuardWithKwargs:
    """Tests for dialect_guard_violations with **kwargs support."""

    def test_no_violations_with_kwargs(self):
        """Test that **kwargs suppresses missing param violations."""
        from src.graph.graph import dialect_guard_violations

        code = "import pandas as pd\ndf = pd.read_csv('data/raw.csv', **dialect)"

        violations = dialect_guard_violations(
            code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8"
        )

        # No violations because **kwargs might supply the params
        assert len(violations) == 0

    def test_literal_mismatch_with_kwargs(self):
        """Test that literal mismatches are still caught with **kwargs."""
        from src.graph.graph import dialect_guard_violations

        code = "import pandas as pd\ndf = pd.read_csv('data/raw.csv', sep=';', **dialect)"

        violations = dialect_guard_violations(
            code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8"
        )

        # sep literal is wrong, should be violation
        assert len(violations) == 1
        assert "sep" in violations[0]

    def test_missing_params_without_kwargs(self):
        """Test that missing params without **kwargs still generate violations."""
        from src.graph.graph import dialect_guard_violations

        code = "import pandas as pd\ndf = pd.read_csv('data/raw.csv')"

        violations = dialect_guard_violations(
            code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8"
        )

        # All 3 params missing
        assert len(violations) == 3

    def test_delimiter_alias_is_accepted_when_matching(self):
        """Test that delimiter alias counts as a matching separator."""
        from src.graph.graph import dialect_guard_violations

        code = "import pandas as pd\ndf = pd.read_csv('data/raw.csv', delimiter=';', decimal=',', encoding='utf-8')"

        violations = dialect_guard_violations(
            code, csv_sep=";", csv_decimal=",", csv_encoding="utf-8"
        )

        assert violations == []
