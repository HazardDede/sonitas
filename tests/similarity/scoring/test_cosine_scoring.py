import numpy as np
import pytest

from sonitas.similarity.scoring import CosineScoring
from sonitas.typestore import Signal


# Fixture for the CosineScoring instance
@pytest.fixture
def cosine_scorer() -> CosineScoring:
    """Returns an instance of CosineScoring."""
    return CosineScoring()


# Test cases for CosineScoring
def test_cosine_identical_vectors(cosine_scorer: CosineScoring):
    """Test cosine similarity with identical non-zero vectors."""
    signal_a: Signal = np.array([1, 2, 3])
    signal_b: Signal = np.array([1, 2, 3])
    assert cosine_scorer.compare(signal_a, signal_b) == pytest.approx(1.0)


def test_cosine_opposite_vectors(cosine_scorer: CosineScoring):
    """Test cosine similarity with opposite non-zero vectors."""
    signal_a: Signal = np.array([1, 2, 3])
    signal_b: Signal = np.array([-1, -2, -3])
    assert cosine_scorer.compare(signal_a, signal_b) == pytest.approx(-1.0)


def test_cosine_orthogonal_vectors(cosine_scorer: CosineScoring):
    """Test cosine similarity with orthogonal non-zero vectors."""
    signal_a: Signal = np.array([1, 0])
    signal_b: Signal = np.array([0, 1])
    assert cosine_scorer.compare(signal_a, signal_b) == pytest.approx(0.0)


def test_cosine_some_similarity(cosine_scorer: CosineScoring):
    """Test cosine similarity with vectors that have some similarity."""
    signal_a: Signal = np.array([1, 2, 3])
    signal_b: Signal = np.array([1, 3, 5]) # Different but in a similar direction
    # Manual calculation:
    # dot = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
    # norm_a = sqrt(1^2 + 2^2 + 3^2) = sqrt(1 + 4 + 9) = sqrt(14)
    # norm_b = sqrt(1^2 + 3^2 + 5^2) = sqrt(1 + 9 + 25) = sqrt(35)
    # expected = 22 / (sqrt(14) * sqrt(35)) = 22 / sqrt(490) = 22 / 22.1359... approx 0.9938...
    expected_similarity = 22 / (np.linalg.norm(signal_a) * np.linalg.norm(signal_b))
    assert cosine_scorer.compare(signal_a, signal_b) == pytest.approx(expected_similarity)


def test_cosine_one_zero_vector(cosine_scorer: CosineScoring):
    """Test cosine similarity when one vector is a zero vector."""
    signal_a: Signal = np.array([1, 2, 3])
    signal_b: Signal = np.array([0, 0, 0])
    assert cosine_scorer.compare(signal_a, signal_b) == 0.0


def test_cosine_both_zero_vectors(cosine_scorer: CosineScoring):
    """Test cosine similarity when both vectors are zero vectors."""
    signal_a: Signal = np.array([0, 0, 0])
    signal_b: Signal = np.array([0, 0, 0])
    assert cosine_scorer.compare(signal_a, signal_b) == 0.0


def test_cosine_floating_point_vectors(cosine_scorer: CosineScoring):
    """Test cosine similarity with vectors containing floating point numbers."""
    signal_a: Signal = np.array([0.1, 0.2, 0.3])
    signal_b: Signal = np.array([0.1, 0.2, 0.3])
    assert cosine_scorer.compare(signal_a, signal_b) == pytest.approx(1.0)

    signal_c: Signal = np.array([0.5, -0.5])
    signal_d: Signal = np.array([-0.5, 0.5])
    assert cosine_scorer.compare(signal_c, signal_d) == pytest.approx(-1.0)


def test_cosine_vectors_with_negative_values(cosine_scorer: CosineScoring):
    """Test cosine similarity with vectors containing negative values."""
    signal_a: Signal = np.array([-1, 2, -3])
    signal_b: Signal = np.array([-1, 2, -3])
    assert cosine_scorer.compare(signal_a, signal_b) == pytest.approx(1.0)

    signal_c: Signal = np.array([1, -2, 3])
    signal_d: Signal = np.array([-2, 4, -6]) # Scaled opposite
    assert cosine_scorer.compare(signal_c, signal_d) == pytest.approx(-1.0)


def test_cosine_single_element_vectors(cosine_scorer: CosineScoring):
    """Test cosine similarity with single element vectors."""
    signal_a: Signal = np.array([5])
    signal_b: Signal = np.array([10])
    assert cosine_scorer.compare(signal_a, signal_b) == pytest.approx(1.0) # Same direction

    signal_c: Signal = np.array([5])
    signal_d: Signal = np.array([-10])
    assert cosine_scorer.compare(signal_c, signal_d) == pytest.approx(-1.0) # Opposite direction

    signal_e: Signal = np.array([0])
    signal_f: Signal = np.array([10])
    assert cosine_scorer.compare(signal_e, signal_f) == 0.0 # One is zero
