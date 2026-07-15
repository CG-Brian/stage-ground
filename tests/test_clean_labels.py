import pytest

from stageground.data.clean_labels import canonicalize


@pytest.mark.parametrize(
    "raw, expected",
    [
        # subtype -> major category
        ("T3a", "T3"),
        ("T1b1", "T1"),
        ("T1b2", "T1"),
        ("T2a2", "T2"),
        ("T4d", "T4"),
        ("N2c", "N2"),
        ("N1a", "N1"),
        ("N3b", "N3"),
        ("M1a", "M1"),
        ("M1c", "M1"),
        # already-major values pass through
        ("T2", "T2"),
        ("N0", "N0"),
        ("M1", "M1"),
        # zero is a real first digit -> preserved (NOT dropped)
        ("T0", "T0"),
        ("N0", "N0"),
        ("M0", "M0"),
        # X = "assessed but indeterminate" -> preserved as-is
        ("TX", "TX"),
        ("NX", "NX"),
        ("MX", "MX"),
        # robustness: whitespace / case
        (" t3a ", "T3"),
        ("pT2", "T2"),  # optional: strip leading 'p' for pathologic prefix
    ],
)
def test_canonicalize(raw, expected):
    assert canonicalize(raw) == expected