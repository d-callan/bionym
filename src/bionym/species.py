"""Organism-name comparison shared by resolver joins and client filters.

Database organism strings are inconsistent: strain suffixes
("Plasmodium falciparum 3D7" vs "Plasmodium falciparum") and abbreviated
genera ("P. falciparum") both appear. `same_species` tolerates both so
joins don't silently miss.
"""

from __future__ import annotations


def species_key(name: str | None) -> str:
    """Genus + species epithet, lowercased — ignores strain/isolate."""
    return " ".join((name or "").lower().split()[:2])


def same_species(a: str | None, b: str | None) -> bool:
    """Species-name match tolerant of strain suffixes and abbreviated
    genera ('P. falciparum' ~ 'Plasmodium falciparum').

    The genus abbreviation is only expanded when one side is actually
    abbreviated (single letter, optionally dotted) — full genus names
    must match exactly, so 'Pseudomonas fluorescens' never joins
    'Proteus fluorescens'.
    """
    ka, kb = species_key(a), species_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    ga, _, ea = ka.partition(" ")
    gb, _, eb = kb.partition(" ")
    if not ea or ea != eb:
        return False
    short, full = (ga, gb) if len(ga.rstrip(".")) <= 1 else (gb, ga)
    initial = short.rstrip(".")
    return len(initial) == 1 and full.startswith(initial)
