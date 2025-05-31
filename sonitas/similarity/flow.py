"""
Defines the signal processing and comparison flow.

This module provides the `Flow` class, which orchestrates a series of
transformations on two input signals and then scores their similarity.
It's designed to be a flexible pipeline where different transformation
steps and scoring methods can be plugged in.
"""
from typing import List

from sonitas.similarity.scoring import Scoring
from sonitas.similarity.transform import Transformer
from sonitas.typestore import Signal


class Flow:
    """
    Represents a processing pipeline for comparing two signals.

    A `Flow` instance takes a list of `Transformer` objects and a `Scoring`
    object. When its `run` method is called with two signals, it applies
    each transformer in sequence to both signals and then uses the scoring
    object to compute a similarity score between the transformed signals.
    """
    def __init__(self, steps: List[Transformer], scoring: Scoring):
        """
        Initializes a new Flow instance.

        Args:
            steps: A list of `Transformer` objects. These will be applied
                   sequentially to the input signals.
            scoring: A `Scoring` object that will be used to compare the
                     signals after all transformations have been applied.
        """
        self.steps = steps
        self.scoring = scoring

    def run(self, signal_a: Signal, signal_b: Signal) -> float:
        """
        Executes the transformation and scoring pipeline on two signals.

        Each transformer in `self.steps` is applied to `signal_a` and
        `signal_b`. The transformed signals are then passed to the
        `self.scoring` object's `compare` method.

        Args:
            signal_a: The first input signal.
            signal_b: The second input signal.

        Returns:
            A float representing the similarity score between the two
            signals after transformation, as determined by the `scoring`
            object.
        """
        for step in self.steps:
            signal_a, signal_b = step.transform(signal_a, signal_b)

        return self.scoring.compare(signal_a, signal_b)
