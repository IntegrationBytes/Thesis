"""Unit tests for pure helpers in pipeline/extract.py that don't touch the
network. (The retry loop and SHACL validation are integration-tested via
the per-vignette runs — only pure helpers go here.)
"""
from __future__ import annotations

from pipeline.extract import strip_markdown_fences


class TestStripMarkdownFences:
    def test_unfenced_input_passes_through(self):
        raw = "@prefix ex: <http://example.org/> .\nex:a ex:b ex:c ."
        assert strip_markdown_fences(raw) == raw

    def test_fenced_turtle_with_language_tag(self):
        raw = "```turtle\n@prefix ex: <http://example.org/> .\nex:a ex:b ex:c .\n```"
        out = strip_markdown_fences(raw)
        assert out.startswith("@prefix")
        assert "```" not in out
        assert "turtle" not in out.splitlines()[0].lower()

    def test_fenced_ttl_language_tag_is_stripped(self):
        raw = "```ttl\n:a :b :c .\n```"
        out = strip_markdown_fences(raw)
        assert out == ":a :b :c ."

    def test_fenced_no_language_tag(self):
        raw = "```\n:a :b :c .\n```"
        out = strip_markdown_fences(raw)
        assert out == ":a :b :c ."

    def test_leading_whitespace_and_trailing_whitespace_are_stripped(self):
        raw = "   \n@prefix ex: <http://example.org/> .\n   "
        assert strip_markdown_fences(raw) == "@prefix ex: <http://example.org/> ."
