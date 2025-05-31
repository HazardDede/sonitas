"""
Signal Transformation Utilities

This module provides a collection of signal transformation classes.
These transformers can be used to preprocess signals before analysis or
comparison. The module defines abstract base classes for unary (single signal)
and binary (two signals) transformations, along with several concrete
implementations.

Key components:
- `BinaryTransform`: Abstract base class for transformations operating on two signals.
- `UnaryTransform`: Abstract base class for transformations operating on a single
  signal, which can also be applied to two signals independently.
- Concrete transformers like `Mixdown`, `Normalize`, `LowPass`, `Magnitude`,
  `PadZero`, and `FFT`.
"""

from abc import abstractmethod, ABCMeta
from typing import Tuple, Union

import numpy as np
from numpy.fft import fft

from sonitas.typestore import Signal

# Type alias for convenience, representing any transformer class instance.
Transformer = Union['UnaryTransform', 'BinaryTransform']


class BinaryTransform(metaclass=ABCMeta):
    """
    Abstract base class for transformations that operate on two signals simultaneously.

    Concrete implementations must provide the `transform` method.
    This is useful for transformations where the processing of one signal
    depends on the other, or where both are modified based on their combined properties
    (e.g., padding to a common length).
    """
    @abstractmethod
    def transform(self, signal_a: Signal, signal_b: Signal) -> Tuple[Signal, Signal]:
        """
        Applies a transformation to two input signals.

        Args:
            signal_a: The first signal to transform.
            signal_b: The second signal to transform.

        Returns:
            A tuple containing the two transformed signals (transformed_a, transformed_b).
        """
        raise NotImplementedError()


class UnaryTransform(BinaryTransform, metaclass=ABCMeta):
    """
    Abstract base class for transformations that operate on a single signal.

    This class inherits from `BinaryTransform` and provides a default
    `transform` method that applies the unary transformation independently
    to two signals. Concrete implementations must provide the `transform_unary`
    method.
    """

    @abstractmethod
    def transform_unary(self, signal: Signal) -> Signal:
        """
        Applies the transformation to a single input signal.

        Args:
            signal: The signal to transform.

        Returns:
            The transformed signal.
        """
        raise NotImplementedError()

    def transform(self, signal_a: Signal, signal_b: Signal) -> Tuple[Signal, Signal]:
        """
        Applies the unary transformation independently to two signals.

        This method calls `transform_unary` for each of the input signals.

        Args:
            signal_a: The first signal.
            signal_b: The second signal.

        Returns:
            A tuple containing the two independently transformed signals.
        """
        return self.transform_unary(signal_a), self.transform_unary(signal_b)


class Mixdown(UnaryTransform):
    """
    Converts a multi-channel signal to a single-channel (mono) signal.

    If the input signal is already mono or 1D, it is returned unchanged.
    For multi-channel signals (e.g., stereo), it computes the mean across
    the channels (typically the second axis for a 2D array like [samples, channels]).
    """
    def transform_unary(self, signal: Signal) -> Signal:
        """
        Applies the mixdown transformation.

        Args:
            signal: The input signal, expected to be a NumPy array.

        Returns:
            The mono (1D) signal.
        """
        if not isinstance(signal, np.ndarray):
            signal = np.asarray(signal)

        chn = len(signal.shape)
        if chn <= 1:
            # Already mono
            return signal
        return signal.mean(axis=1)


class Normalize(UnaryTransform):
    """
    Normalizes a signal to have a mean of 0 and a standard deviation of 1.

    This is also known as Z-score normalization. If the signal's standard
    deviation is below a small epsilon value (to prevent division by zero
    or near-zero), the signal is transformed to all zeros.
    """
    def __init__(self, eps: float = 1e-10):
        """
        Initializes the Normalize transformer.

        Args:
            eps: A small epsilon value to prevent division by zero or
                 instability when the standard deviation is very small.
                 If std < eps, the signal becomes all zeros.
        """
        self.eps = float(eps)

    def transform_unary(self, signal: Signal) -> Signal:
        """
        Applies Z-score normalization to the signal.

        Args:
            signal: The input signal (NumPy array).

        Returns:
            The normalized signal.
        """
        std = np.std(signal)
        if std < self.eps:
            # Signal is constant or near-constant.
            # Returning zeros is one way to handle this.
            # Another might be to return the signal as is if mean is also ~0,
            # or just signal - mean.
            return signal * 0  # or raise warning
        return (signal - np.mean(signal)) / std


class LowPass(UnaryTransform):
    """
    Applies a simple low-pass filter by truncating the higher frequency
    components in the signal's spectrum.

    This transformer operates directly on the input signal, assuming it is
    already in a domain (e.g., FFT magnitude spectrum) where truncation
    equates to low-pass filtering. It keeps a specified fraction of the
    initial components.

    Note: This is a very basic form of low-pass filtering. For signals in the
    time domain, one would typically apply an FFT, zero out high-frequency bins,
    and then apply an inverse FFT, or use a time-domain filter design.
    This class assumes the input `signal` is something like an FFT magnitude array.
    """
    def __init__(self, keep_ratio: float = 0.5):
        """
        Initializes the LowPass transformer.

        Args:
            keep_ratio: The fraction of the signal's components (presumably
                        ordered by frequency) to keep. Must be between 0.0 and 1.0.
                        For example, 0.5 keeps the first half of the components.
        """
        if not 0.0 <= keep_ratio <= 1.0:
            raise ValueError("`keep_ratio` must be between 0 and 1")
        self.keep_ratio = float(keep_ratio)

    def transform_unary(self, signal: Signal) -> Signal:
        if signal.ndim == 0:  # Handle scalar input
            return signal

        cutoff = int(len(signal) * self.keep_ratio)
        # Ensure at least one element if ratio > 0 and signal not empty
        if cutoff == 0 and self.keep_ratio > 0 and len(signal) > 0:
            # Ensure at least one element if ratio > 0 and signal not empty
            cutoff = 1
        return signal[:cutoff]


class Magnitude(UnaryTransform):
    """
    Computes the element-wise magnitude (absolute value) of the signal.

    This is often used after an FFT to get the magnitude spectrum from
    complex FFT coefficients.
    """
    def transform_unary(self, signal: Signal) -> Signal:
        """
        Calculates the absolute value of each element in the signal.

        Args:
            signal: The input signal, which can contain complex numbers.

        Returns:
            A signal containing the magnitudes.
        """
        return np.abs(signal)


class PadZero(BinaryTransform):
    """
    Pads two signals with zeros so they both have the same length.

    The target length is the smallest power of two greater than or equal to
    the length of the longer signal. This is often done to optimize FFT
    computations.
    """
    def transform(self, signal_a: Signal, signal_b: Signal) -> Tuple[Signal, Signal]:
        """
        Pads both signals with trailing zeros to a common power-of-two length.

        Args:
            signal_a: The first signal.
            signal_b: The second signal.

        Returns:
            A tuple containing the two padded signals.
        """
        max_len = max(len(signal_a), len(signal_b))
        if max_len == 0:  # Handle empty signals
            return signal_a, signal_b

        # Calculate the next power of two
        # (unless max_len is already a power of two)
        if max_len & (max_len - 1) == 0 and max_len != 0:  # Check if power of two
            n = max_len
        else:
            n = int(2 ** np.ceil(np.log2(max_len)))

        return (
            np.pad(signal_a, (0, n - len(signal_a)), mode='constant', constant_values=0),
            np.pad(signal_b, (0, n - len(signal_b)), mode='constant', constant_values=0)
        )


class FFT(UnaryTransform):
    """
    Performs a Fast Fourier Transform (FFT) on the signal.

    This transforms a time-domain signal into its frequency-domain representation.
    The result contains complex numbers representing magnitude and phase.
    """
    def transform_unary(self, signal: Signal) -> Signal:
        """
        Applies the FFT to the input signal.

        Args:
            signal: The input time-domain signal.

        Returns:
            The complex-valued frequency-domain representation of the signal.
        """
        return fft(signal)
