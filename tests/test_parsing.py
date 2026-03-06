"""Tests for ra.utils.parsing — final-answer extraction and code-block finding."""

from __future__ import annotations

import json


from ra.utils.parsing import find_code_blocks, find_final_answer


# ---------------------------------------------------------------------------
# find_final_answer — FINAL(...)
# ---------------------------------------------------------------------------


class TestFinalSimple:
    def test_plain_string(self) -> None:
        assert find_final_answer("FINAL(hello world)") == "hello world"

    def test_strips_whitespace(self) -> None:
        assert find_final_answer("FINAL(  answer  )") == "answer"

    def test_multiline_content(self) -> None:
        text = "FINAL(line1\nline2\nline3)"
        assert find_final_answer(text) == "line1\nline2\nline3"

    def test_none_when_absent(self) -> None:
        assert find_final_answer("no marker here") is None

    def test_must_be_at_line_start(self) -> None:
        assert find_final_answer("some text FINAL(x)") is None

    def test_indented_is_ok(self) -> None:
        assert find_final_answer("  FINAL(ok)") == "ok"


class TestFinalNestedParens:
    """The old regex ``FINAL\\((.*?)\\)`` truncated on the first ``)``."""

    def test_json_with_function_names(self) -> None:
        """Regression: JSON containing ``toJSON()`` was truncated."""
        payload = json.dumps(
            [
                {
                    "hypothesis": "Prototype pollution",
                    "exploit_sketch": (
                        "1. Parse YAML\n"
                        "2. YAMLMap.toJSON() calls addPairToJSMap()\n"
                        "3. Pollution succeeds"
                    ),
                }
            ]
        )
        text = f"FINAL({payload})"
        result = find_final_answer(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed[0]["exploit_sketch"].endswith("Pollution succeeds")

    def test_nested_parens(self) -> None:
        text = "FINAL(foo(bar(baz)))"
        assert find_final_answer(text) == "foo(bar(baz))"

    def test_single_nested_paren(self) -> None:
        text = "FINAL(addPairToJSMap())"
        assert find_final_answer(text) == "addPairToJSMap()"

    def test_multiple_parens_in_json(self) -> None:
        payload = json.dumps(
            {"a": "fn()", "b": "g(h())", "c": "plain"}
        )
        text = f"FINAL({payload})"
        result = find_final_answer(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["b"] == "g(h())"

    def test_unbalanced_returns_none(self) -> None:
        text = "FINAL(open paren ( but never closed"
        assert find_final_answer(text) is None


# ---------------------------------------------------------------------------
# find_final_answer — FINAL_VAR(...)
# ---------------------------------------------------------------------------


class TestFinalVar:
    def test_returns_none_without_environment(self) -> None:
        assert find_final_answer("FINAL_VAR(result)") is None

    def test_with_quotes(self) -> None:
        # FINAL_VAR("result") should strip quotes for the variable name
        assert find_final_answer('FINAL_VAR("result")') is None

    def test_must_be_at_line_start(self) -> None:
        assert find_final_answer("text FINAL_VAR(x)") is None


# ---------------------------------------------------------------------------
# find_code_blocks
# ---------------------------------------------------------------------------


class TestFindCodeBlocks:
    def test_single_block(self) -> None:
        text = "text\n```repl\nprint('hi')\n```\nmore"
        blocks = find_code_blocks(text)
        assert blocks == ["print('hi')"]

    def test_multiple_blocks(self) -> None:
        text = "```repl\na = 1\n```\ntext\n```repl\nb = 2\n```"
        assert find_code_blocks(text) == ["a = 1", "b = 2"]

    def test_no_blocks(self) -> None:
        assert find_code_blocks("nothing here") == []

    def test_wrong_language_ignored(self) -> None:
        text = "```python\nprint(1)\n```"
        assert find_code_blocks(text) == []
