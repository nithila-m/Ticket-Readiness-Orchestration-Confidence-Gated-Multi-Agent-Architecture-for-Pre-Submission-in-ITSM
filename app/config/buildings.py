"""
Building-name normalisation shared by Agent 4 (correlation), its seed
ticket pool and the dataset augmentation script.

The synthetic dataset spells some buildings two ways ("Men's Hostel A" /
"Men's Hostel A-Block", "Ladies Hostel D" / "Ladies Hostel D-Block"), so
building equality must be checked on the normalised form.
"""

import re

_BLOCK_SUFFIX = re.compile(r"[\s-]*block$")


def normalize_building(name: str | None) -> str | None:
    """Lower-cased, whitespace-collapsed, trailing "-Block" removed; None if empty."""
    if name is None:
        return None
    key = " ".join(str(name).lower().split())
    key = _BLOCK_SUFFIX.sub("", key).strip()
    return key or None
