"""Staged JEV question builders.

Hard rules (see plan):
- Questions are Python-templated only; no LLM generates questions or options.
- `criteria` options always come from API result sets.
- Every Choice question includes a `none`/`other` escape option.
- JEV judges over handed evidence; it never enumerates possibilities.
"""
