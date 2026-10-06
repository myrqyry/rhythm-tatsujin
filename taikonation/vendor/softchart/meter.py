"""Exact rational-meter helpers for the 96-slot-per-4/4 TJA lattice."""

from fractions import Fraction
from math import gcd
from numbers import Real

import numpy as np

from .vocab import SLOTS, WINDOW


def _expand(value, n_measures, name, dtype):
    array = np.asarray(value)
    if array.ndim == 0:
        array = np.full(n_measures, array.item())
    else:
        array = array.reshape(-1)
        if len(array) != n_measures:
            raise ValueError(
                f"{name} must contain one value per measure "
                f"({n_measures}), got {len(array)}"
            )
    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(dtype)
        values = []
        for raw in array.flat:
            if isinstance(raw, (bool, np.bool_)):
                raise ValueError(f"{name} must contain finite integers")
            if isinstance(raw, (int, np.integer)):
                integer = int(raw)
            elif isinstance(raw, (float, np.floating)):
                number = float(raw)
                if not np.isfinite(number) or not number.is_integer():
                    raise ValueError(f"{name} must contain finite integers")
                integer = int(number)
            else:
                raise ValueError(f"{name} must contain finite integers")
            if not info.min <= integer <= info.max:
                raise ValueError(f"{name} is outside the {np.dtype(dtype)} range")
            values.append(integer)
        return np.asarray(values, dtype=dtype).reshape(array.shape)
    values = []
    for raw in array.flat:
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Real):
            raise ValueError(f"{name} must contain numeric values")
        values.append(float(raw))
    return np.asarray(values, dtype=dtype).reshape(array.shape)


def rational_meter_arrays(measure_num, measure_den, n_measures=None):
    """Validate TJA ``#MEASURE`` fractions and return lattice properties.

    Returns ``(slots, numerator, denominator, quarter_beats)``. A meter is
    representable exactly iff ``96 * numerator / denominator`` is an integer;
    no rounding or coercion is permitted.
    """
    num = np.asarray(measure_num)
    den = np.asarray(measure_den)
    if n_measures is None:
        if num.ndim == 0 and den.ndim == 0:
            n_measures = 1
        else:
            n_measures = max(num.size, den.size)
    num = _expand(num, n_measures, "measure_num", np.int64)
    den = _expand(den, n_measures, "measure_den", np.int64)
    if (num <= 0).any() or (den <= 0).any():
        raise ValueError("measure numerator and denominator must be positive")

    slot_values = []
    for numerator, denominator in zip(num.tolist(), den.tolist()):
        common = gcd(int(numerator), int(denominator))
        reduced_num = int(numerator) // common
        reduced_den = int(denominator) // common
        if SLOTS % reduced_den:
            raise ValueError(
                "meter is not exactly representable on the 96-slot lattice")
        slot_count = (SLOTS // reduced_den) * reduced_num
        if not 1 <= slot_count <= WINDOW:
            raise ValueError(
                f"measure lattice length must be within 1..{WINDOW} slots")
        slot_values.append(slot_count)
    slots = np.asarray(slot_values, dtype=np.int64)
    quarter_beats = 4.0 * num / den
    return slots, num, den, quarter_beats.astype(np.float64)


def resolve_meter_layout(grid, n_measures):
    """Resolve exact per-measure layout from a generation grid.

    Preferred input is ``measure_num``/``measure_den`` with optional
    ``measure_slots`` for an integrity check. A uniform ``meter`` or a
    ``beats_per_measure`` array remains a supported grid API: each value is a
    quarter-note count and is converted to an exact reduced TJA fraction.
    """
    has_num = "measure_num" in grid
    has_den = "measure_den" in grid
    if has_num != has_den:
        raise ValueError("measure_num and measure_den must be provided together")
    if has_num:
        num = _expand(grid["measure_num"], n_measures, "measure_num", np.int64)
        den = _expand(grid["measure_den"], n_measures, "measure_den", np.int64)
    else:
        raw = grid.get("beats_per_measure", grid.get("meter"))
        if raw is None:
            raise ValueError(
                "slot generation requires measure_num/measure_den or an "
                "explicit uniform meter/beats_per_measure grid"
            )
        quarter = _expand(raw, n_measures, "beats_per_measure", np.float64)
        if not np.isfinite(quarter).all() or (quarter <= 0).any():
            raise ValueError("quarter beats per measure must be positive and finite")
        fractions = [Fraction(float(q)).limit_denominator(768) for q in quarter]
        if any(abs(float(f) - q) > 1e-9
               for f, q in zip(fractions, quarter)):
            raise ValueError("quarter-beat meter values must be exact rationals")
        num = np.asarray([f.numerator for f in fractions], dtype=np.int64)
        den = np.asarray([4 * f.denominator for f in fractions], dtype=np.int64)

    slots, num, den, quarter = rational_meter_arrays(num, den, n_measures)
    if "measure_slots" in grid:
        supplied = _expand(
            grid["measure_slots"], n_measures, "measure_slots", np.int64)
        if not np.array_equal(supplied, slots):
            raise ValueError("measure_slots disagrees with measure_num/measure_den")
    return slots, num, den, quarter
