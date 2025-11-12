import pytest
from auto_project.validators import not_empty, no_nulls, unique, positive_numbers, required_keys

def test_not_empty_ok():
    not_empty([1])

def test_not_empty_fail():
    with pytest.raises(AssertionError):
        not_empty([])

def test_no_nulls():
    no_nulls([1,2,3])
    with pytest.raises(AssertionError):
        no_nulls([1,None,3])

def test_unique():
    unique([1,2,3])
    with pytest.raises(AssertionError):
        unique([1,1,2])

def test_positive_numbers():
    positive_numbers([0.1, 2, 3])
    with pytest.raises(AssertionError):
        positive_numbers([1, -2, 3])

def test_required_keys():
    required_keys({"a":1,"b":2}, ["a","b"])
    with pytest.raises(AssertionError):
        required_keys({"a":1}, ["a","b"])
