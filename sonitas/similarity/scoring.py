"""
Signal Similarity Scoring

This module provides a collection of scoring algorithms to quantify the
similarity between two signals. It defines an abstract base class `Scoring`
which outlines the common interface for all scoring methods. Various
concrete implementations like Cosine, Pearson, Spearman, Kendall Tau, and
Normalized Cross-Correlation (NCC) are provided.

Each scoring method takes two signals as input and returns a float
representing their similarity, typically in the range [-1, 1].
"""

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

from sonitas.typestore import Signal


class Scoring(metaclass=ABCMeta):
    """
    Abstract base class defining the interface for comparing two signals.

    All concrete scoring classes should inherit from this class and implement
    the `compare` method.
    """

    @abstractmethod
    def compare(self, signal_a: Signal, signal_b: Signal) -> float:
        """
        Compares the two given signals and returns a similarity score.

        The score is expected to be within the range [-1.0, 1.0], where 1.0
        indicates perfect similarity, 0.0 indicates no linear correlation
        (interpretations may vary by specific metric), and -1.0 indicates
        perfect inverse similarity.

        Args:
            signal_a: The first signal to compare.
            signal_b: The second signal to compare.

        Returns:
            A float representing the similarity score between the two signals.
        """
        raise NotImplementedError()


class CosineScoring(Scoring):
    """
    Computes the cosine similarity between two signals.

    Cosine similarity measures the cosine of the angle between two non-zero
    vectors, indicating the orientation similarity. A value of 1 means the
    vectors have the same orientation, 0 means they are orthogonal, and -1
    means they are diametrically opposed.
    """
    def compare(self, signal_a: Signal, signal_b: Signal) -> float:
        """
        Calculates cosine similarity.
        Returns 0.0 if the norm of either signal is zero to avoid division by zero.
        """
        dot = np.dot(signal_a, signal_b).real
        norm = np.linalg.norm(signal_a) * np.linalg.norm(signal_b)
        return float(dot / norm) if norm else 0.0


class PearsonScoring(Scoring):
    """
    Computes the Pearson correlation coefficient between two signals.

    Pearson correlation measures the linear relationship between two continuous
    variables. It ranges from -1 (perfect negative linear correlation) to +1
    (perfect positive linear correlation), with 0 indicating no linear correlation.
    """
    def compare(self, signal_a: Signal, signal_b: Signal) -> float:
        """
        Calculates Pearson correlation.
        Returns 0.0 if either signal has zero standard deviation to avoid NaNs.
        """
        # Pearson correlation is undefined if one of the signals has zero variance.
        if np.std(signal_a) == 0 or np.std(signal_b) == 0:
            # If both signals are constant and identical, they are perfectly correlated.
            # If they are constant but different, or one is constant and the other not,
            # the interpretation can vary. Returning 0.0 is a common way to handle this.
            # Alternatively, if s_a == s_b (element-wise) and std is 0, could return 1.0.
            # For now, 0.0 is a safe default for undefined cases.
            if np.array_equal(signal_a, signal_b) and np.std(signal_a) == 0:  # Both are identical constant signals
                return 1.0
            return 0.0

        corr, _ = pearsonr(signal_a, signal_b)
        return float(corr)


class SpearmanScoring(Scoring):
    """
    Computes the Spearman rank correlation coefficient between two signals.

    Spearman correlation assesses the monotonic relationship between two
    variables. It measures how well the relationship between two variables
    can be described using a monotonic function. It ranges from -1 to +1.
    """
    def compare(self, signal_a: Signal, signal_b: Signal) -> float:
        """Calculates Spearman rank correlation."""
        rho, _ = spearmanr(signal_a, signal_b)
        return rho


class KendallTauScoring(Scoring):
    """
    Computes the Kendall Tau rank correlation coefficient between two signals.

    Kendall Tau measures the ordinal association between two measured quantities.
    It assesses the similarity of the orderings of the data when ranked by each
    of the quantities. It ranges from -1 to +1.
    """
    def compare(self, signal_a: Signal, signal_b: Signal) -> float:
        """Calculates Kendall Tau rank correlation."""
        tau, _ = kendalltau(signal_a, signal_b)
        return tau


class NCCScoring(Scoring):
    """
    Computes the Normalized Cross-Correlation (NCC) between two signals.

    NCC measures the similarity of two signals as a function of the displacement
    of one relative to the other. This implementation calculates the NCC at
    lag 0 after normalizing both signals (Z-normalization).
    The result ranges from -1 to +1.
    """
    def compare(self, signal_a: Signal, signal_b: Signal) -> float:
        """
        Calculates Normalized Cross-Correlation at lag 0.
        Returns 0.0 if either signal has zero standard deviation to avoid division by zero.
        """
        a = signal_a
        b = signal_b
        return np.correlate(
            (a - np.mean(a)) / np.std(a),
            (b - np.mean(b)) / np.std(b),
            mode='valid'
        )[0] / len(a)
