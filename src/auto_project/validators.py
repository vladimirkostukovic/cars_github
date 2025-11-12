from __future__ import annotations
from typing import Mapping, Sequence, Any, Iterable

def not_empty(value: Iterable[Any], name: str = "value") -> None:
    if value is None:
        raise AssertionError(f"{name} is None")
    try:
        if len(value) == 0:  # type: ignore[func-returns-value]
            raise AssertionError(f"{name} is empty")
    except TypeError:
        if not any(True for _ in value):
            raise AssertionError(f"{name} produced no items")

def no_nulls(seq: Sequence[Any], name: str = "seq") -> None:
    if any(x is None for x in seq):
        raise AssertionError(f"{name} contains nulls")

def unique(seq: Sequence[Any], name: str = "seq") -> None:
    if len(set(seq)) != len(seq):
        raise AssertionError(f"{name} must be unique")

def positive_numbers(seq: Sequence[float | int], name: str = "seq") -> None:
    if any((x is None) or (float(x) <= 0) for x in seq):  # type: ignore[arg-type]
        raise AssertionError(f"{name} must contain positive numbers only")

def required_keys(obj: Mapping[str, Any], keys: Sequence[str], name: str = "mapping") -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise AssertionError(f"{name} missing keys: {', '.join(missing)}")
