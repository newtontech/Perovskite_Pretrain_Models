"""Deterministic molecular-generation and screening scaffolds."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from perovskite_pretrain.property_prediction import DeterministicPropertyPredictor


ATOM_PATTERN = re.compile(r"Cl|Br|[A-Z][a-z]?|[cnops]")
BALANCED_PAIRS = {"(": ")", "[": "]"}


@dataclass(frozen=True)
class Candidate:
    smiles: str
    source_seed_smiles: str
    generation_method: str = "deterministic_substituent_append"


@dataclass(frozen=True)
class EvaluatedCandidate:
    smiles: str
    source_seed_smiles: str
    generation_method: str
    valid: bool
    novel: bool
    duplicate: bool
    predictions: dict[str, float] = field(default_factory=dict)
    filter_notes: tuple[str, ...] = ()


class CandidateGenerator:
    """Generate small deterministic candidate sets from seed SMILES."""

    def __init__(self, seed: int = 0, substituents: Iterable[str] | None = None):
        self.seed = seed
        self.substituents = tuple(substituents or ("N", "O", "F", "Cl", "C#N"))

    def generate(self, seed_smiles: Iterable[str], n_per_seed: int = 5) -> list[Candidate]:
        if n_per_seed <= 0:
            raise ValueError("n_per_seed must be positive")

        candidates: list[Candidate] = []
        seen: set[str] = set()
        for smiles in seed_smiles:
            stripped = smiles.strip()
            if not stripped:
                continue
            for substituent in self.substituents[:n_per_seed]:
                candidate_smiles = f"{stripped}{substituent}"
                if candidate_smiles in seen:
                    continue
                seen.add(candidate_smiles)
                candidates.append(
                    Candidate(smiles=candidate_smiles, source_seed_smiles=stripped)
                )
        return candidates


def validate_smiles_syntax(
    smiles: str, allowed_atoms: set[str] | None = None
) -> tuple[bool, tuple[str, ...]]:
    """Perform dependency-free structural checks before optional RDKit screening."""

    notes: list[str] = []
    if not smiles:
        notes.append("empty_smiles")

    for open_char, close_char in BALANCED_PAIRS.items():
        if smiles.count(open_char) != smiles.count(close_char):
            notes.append(f"unbalanced_{open_char}{close_char}")

    atoms = [match.group(0).capitalize() for match in ATOM_PATTERN.finditer(smiles)]
    if not atoms:
        notes.append("no_atoms_detected")

    if allowed_atoms is not None:
        disallowed = sorted({atom for atom in atoms if atom not in allowed_atoms})
        if disallowed:
            notes.append("disallowed_atoms:" + ",".join(disallowed))

    return not notes, tuple(notes)


def evaluate_candidates(
    candidates: Iterable[Candidate],
    training_smiles: Iterable[str],
    predictor: DeterministicPropertyPredictor | None = None,
    allowed_atoms: set[str] | None = None,
) -> list[EvaluatedCandidate]:
    """Validate, de-duplicate, check novelty, and optionally score candidates."""

    training_set = {smiles.strip() for smiles in training_smiles if smiles.strip()}
    seen: set[str] = set()
    evaluated: list[EvaluatedCandidate] = []

    for candidate in candidates:
        duplicate = candidate.smiles in seen
        seen.add(candidate.smiles)
        valid, notes = validate_smiles_syntax(candidate.smiles, allowed_atoms=allowed_atoms)
        novel = candidate.smiles not in training_set
        all_notes = list(notes)
        if duplicate:
            all_notes.append("duplicate_candidate")
        if not novel:
            all_notes.append("seen_in_training")

        predictions: dict[str, float] = {}
        if predictor is not None and valid:
            predictions = predictor.predict_smiles(candidate.smiles)

        evaluated.append(
            EvaluatedCandidate(
                smiles=candidate.smiles,
                source_seed_smiles=candidate.source_seed_smiles,
                generation_method=candidate.generation_method,
                valid=valid,
                novel=novel,
                duplicate=duplicate,
                predictions=predictions,
                filter_notes=tuple(all_notes),
            )
        )

    return evaluated


def summarize_screening(evaluated: Iterable[EvaluatedCandidate]) -> dict[str, float | int]:
    rows = list(evaluated)
    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "valid": 0,
            "novel": 0,
            "duplicates": 0,
            "valid_fraction": 0.0,
            "novel_fraction": 0.0,
            "duplicate_fraction": 0.0,
        }

    valid = sum(row.valid for row in rows)
    novel = sum(row.novel for row in rows)
    duplicates = sum(row.duplicate for row in rows)
    return {
        "total": total,
        "valid": valid,
        "novel": novel,
        "duplicates": duplicates,
        "valid_fraction": valid / total,
        "novel_fraction": novel / total,
        "duplicate_fraction": duplicates / total,
    }
