"""Shared numeric contract for mixed instance supervision."""

from __future__ import annotations


DENSE_MASK = 0
BOX = 1
POINT = 2

MODE_TO_ID = {
    "dense_mask": DENSE_MASK,
    "box": BOX,
    "point": POINT,
}
ID_TO_MODE = {value: key for key, value in MODE_TO_ID.items()}


def supervision_mode_id(mode: str | None) -> int:
    # Queries from the legacy instance dataset have dense masks and predate the
    # explicit supervision field.
    if mode is None:
        return DENSE_MASK
    try:
        return MODE_TO_ID[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown instance supervision mode: {mode}") from exc
