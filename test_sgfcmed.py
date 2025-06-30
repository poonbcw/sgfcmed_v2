"""
Unit tests for SGFCMedParallel string clustering algorithm.

This test suite covers:
- Internal Levenshtein distance correctness
- Initialization parameters
- Fitting process and prototype selection
- Membership matrix validity (rows sum to 1)
- Prediction correctness and stability
- Edge cases such as more clusters than data or identical strings
- Convergence and reproducibility
"""

import pytest
from sgfcmed_parallel import SGFCMedParallel
from pytest import approx
import random

@pytest.fixture
def model():
    """
    Fixture that returns a default SGFCMedParallel model with 2 clusters.
    """
    return SGFCMedParallel(C=2, m=2.0, max_iter=100, tol=1e-4)

@pytest.fixture
def data():
    """
    Fixture providing a simple dataset of short similar strings.
    """
    return ["book", "back", "boon", "cook", "look", "cool", "kick", "lack", "rack", "tack"]

def test_levenshtein_distance(model,data):
    """
    Test the internal _levenshtein method.

    Verifies that:
    - identical strings return distance 0
    - single char changes return distance 1
    - disjoint strings return distance equal to length
    - empty vs non-empty returns length of non-empty string
    """
    lev = model._levenshtein
    assert lev(data[0], data[0]) == 0
    assert lev(data[0], data[1]) == 2
    assert lev(data[0], data[2]) == 1
    assert lev(data[0], data[9]) == 3
    assert lev("", "abc") == 3
    assert lev("abc", "") == 3
    assert lev("", "") == 0

def test_initialization(model):
    """
    Ensure that model initialization correctly sets parameters.
    """
    assert model.C == 2
    assert model.m == 2.0
    assert model.max_iter == 100
    assert model.tol == 1e-4

def test_fit_and_prototypes(model, data):
    """
    Ensure that after fitting, prototypes are correctly selected
    from generated candidate sets.

    Verifies that:
    - The model returns exactly C prototypes.
    - Each prototype chosen for a cluster appears in the candidate
      lists generated during the algorithm's internal selection process.
    """
    model.fit(data)
    prototypes = model.prototypes()
    assert len(prototypes) == 2

    for i, proto in enumerate(prototypes):
        candidate_rounds = model.get_debug_candidates(i)
        flat_candidates = [c for round_candidates in candidate_rounds for c in round_candidates]
        assert proto in flat_candidates, f"Prototype {proto} not in candidate list for cluster {i}"


def test_prototype_has_minimum_distance(model, data):
    """
    Test that each selected prototype minimizes the total distance
    among its candidate set.

    Verifies that:
    - For each cluster, the chosen prototype has the smallest total
      Levenshtein distance to all data points compared to all
      generated candidate strings.
    """
    model.fit(data)
    prototypes = model.prototypes()

    for i, proto in enumerate(prototypes):
        candidate_rounds = model.get_debug_candidates(i)
        flat_candidates = [c for round in candidate_rounds for c in round]

        proto_dist = sum(model._levenshtein(proto, s) for s in data)
        all_distances = [(cand, sum(model._levenshtein(cand, s) for s in data)) for cand in flat_candidates]
        best_candidate, min_dist = min(all_distances, key=lambda x: x[1])

        assert proto_dist == min_dist, f"Prototype for cluster {i} is not optimal. Best: {best_candidate}, Got: {proto}"


def test_membership_matrix(model, data):
    """
    Test that the membership matrix U has the correct shape
    and rows sum approximately to 1 (due to fuzzy membership).
    """
    model.fit(data)
    U = model.membership()
    assert len(U) == len(data)
    for row in U:
        assert len(row) == 2
        assert abs(sum(row) - 1.0) < 1e-3

def test_predict(model, data):
    """
    Ensure predict returns a valid cluster index for each new string.
    """
    model.fit(data)
    preds = model.predict(["hack", "rook", "cook"])
    assert len(preds) == 3
    for p in preds:
        assert p in [0, 1]

def test_invalid_cluster_count():
    """
    Fitting with more clusters than unique data points should raise ValueError.
    """
    with pytest.raises(ValueError):
        SGFCMedParallel(C=10).fit(["a", "b", "c"])

def test_convergence():
    """
    Verify model converges on a small dataset and the membership
    matrix shape is valid.
    """
    short_data = ["aaa", "aab", "aba", "abb", "bbb"]
    model = SGFCMedParallel(C=2, max_iter=100, tol=0.001)
    model.fit(short_data)
    assert len(model.prototypes()) <= 2
    U = model.membership()
    for row in U:
        assert len(row) == 2

def test_predict_stability():
    """
    Calling predict multiple times on the same data after fit
    should produce identical results.
    """
    input_data = ["abc", "abd", "bbc"]
    model = SGFCMedParallel(C=2)
    model.fit(input_data)
    preds1 = model.predict(input_data)
    preds2 = model.predict(input_data)
    assert preds1 == preds2

def test_identical_strings():
    """
    With identical strings, model should still produce valid predictions
    and likely assign all to the same cluster.
    """
    data = ["aaa", "aaa", "aaa", "aaa"]
    model = SGFCMedParallel(C=2, m=2.0, max_iter=50)
    model.fit(data)
    preds = model.predict(data)
    assert all(p == preds[0] for p in preds)

def test_varied_strings():
    """
    On varied strings, check that membership matrix rows still sum to ~1.
    """
    data = ["apple", "banana", "carrot", "date", "eggplant"]
    model = SGFCMedParallel(C=3, m=2.0, max_iter=100)
    model.fit(data)
    memberships = model.membership()
    assert len(memberships) == len(data)
    assert all(abs(sum(u) - 1.0) < 1e-6 for u in memberships)

def test_stable_prototypes():
    """
    Ensure that fitting twice on the same data produces stable prototypes.
    """
    data = ["kitten", "sitting", "biting", "mitten"]
    model1 = SGFCMedParallel(C=2, m=2.0, max_iter=100, tol=0.00001)
    model1.fit(data)
    proto1 = model1.prototypes()
    model2 = SGFCMedParallel(C=2, m=2.0, max_iter=100, tol=0.00001)
    model2.fit(data)
    proto2 = model2.prototypes()
    assert proto1 == proto2


def test_predict_returns_valid_indices():
    """
    Predict on training data should return valid cluster indices within range.
    """
    data = ["red", "green", "blue", "yellow"]
    model = SGFCMedParallel(C=2)
    model.fit(data)
    preds = model.predict(data)
    assert all(isinstance(i, int) and 0 <= i < 2 for i in preds)

def test_single_data_point():
    """
    Ensure model correctly handles a single data point.

    Verifies that:
    - The prototype is exactly the same as the input.
    - The membership matrix has one row and assigns full membership (1.0)
      to the only available cluster.
    """
    data = ["singleton"]
    model = SGFCMedParallel(C=1)
    model.fit(data)
    prototypes = model.prototypes()
    assert prototypes == data
    membership = model.membership()
    assert len(membership) == 1
    assert membership[0][0] == pytest.approx(1.0)


def test_fit_empty_data_raises():
    """
    Test that fitting on an empty dataset raises a ValueError.

    Ensures that:
    - Calling fit([]) triggers a ValueError, enforcing that data must be non-empty.
    """
    model = SGFCMedParallel(C=2)
    with pytest.raises(ValueError):
        model.fit([])