"""Turtle output sanitiser for LLM-emitted IRIs that violate Turtle local-name grammar.

gpt-oss-120b (and occasionally other LLMs) sometimes produce prefix-shortened
IRIs whose local name contains characters disallowed by the Turtle grammar —
most commonly `+` and additional `:` (typically from ISO-8601 timezone
suffixes such as `ex:visit_2019_03_15+01:00`). The sanitiser substitutes:

  '+' -> 'p'        (e.g. ``ex:visit_2019_03_15p01_00``)
  ':' -> '_'        (only the *additional* ':' after the prefix delimiter)

inside the prefix-shortened part of affected IRIs. The substitution is
deterministic, idempotent on already-valid Turtle, and preserves semantics
(the substituted IRIs are still distinct named individuals).

Used by ``evaluate.py`` as a fallback when an output fails to parse, and
optionally as a post-extraction pass on saved outputs. See §IV-B of the
thesis for an empirical evaluation (91% salvage rate on parse failures
for gpt-oss-120b at n=200).
"""
from __future__ import annotations
import re

_LOCAL_NAME_RE = re.compile(
    r"\b(ex:|chr:|sulo:|rdf:|rdfs:|owl:)([A-Za-z0-9_\-/.~+:%]+?)(?=[\s;,.\)\]]|$)"
)


def sanitize_ttl(text: str) -> str:
    """Replace `+` -> `p` and additional `:` -> `_` in IRI local names.

    Idempotent: applying this function to valid Turtle returns the same text.
    """
    def _fix(m: re.Match) -> str:
        return f"{m.group(1)}{m.group(2).replace('+', 'p').replace(':', '_')}"
    return _LOCAL_NAME_RE.sub(_fix, text)
