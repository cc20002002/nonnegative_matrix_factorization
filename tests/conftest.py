"""Configuration module for tests."""
import pytest

@pytest.fixture(params=[-1, 0, 1])
def integers(request):
    return request.param

