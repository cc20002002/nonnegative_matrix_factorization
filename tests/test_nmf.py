"""
Tests for `nmf` module.
"""

from nmf import nmf

def test_is_positive(integers):
    result = nmf.is_positive(integers)
    true_result = integers > 0
    assert (result == true_result)
