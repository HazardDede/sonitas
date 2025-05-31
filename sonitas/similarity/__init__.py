"""
Sonitas Similarity Package Initialization.

This package provides tools and utilities for comparing signals, primarily
focusing on various scoring mechanisms and transformation steps that can be
applied to signals before comparison.

This `__init__.py` file makes key components like scoring and transform
modules directly accessible. It also defines default configurations and
mappings for supported scoring algorithms and signal transformers.

Attributes:
    DEFAULT_SCORING (str): The default scoring algorithm to be used if none is
        specified. Currently set to 'cosine'.
    SUPPORTED_SCORING (dict): A dictionary mapping string identifiers to their
        respective scoring class implementations from the `scoring` module.
        This allows for easy instantiation of scoring objects by name.
    SUPPORTED_TRANSFORMER (dict): A dictionary mapping string identifiers to
        their respective transformer class implementations from the `transform`
        module. This facilitates the creation of transformation pipelines.
"""
from . import scoring
from . import transform


# The default scoring method to be used when comparing signals if not
# explicitly specified.
DEFAULT_SCORING = 'cosine'


# A registry of supported scoring algorithms.
# Keys are human-readable names (strings) for the algorithms,
# and values are the corresponding scoring class constructors.
# This allows for dynamic selection and instantiation of scoring methods.
SUPPORTED_SCORING = {
    'cosine': scoring.CosineScoring,        # Cosine similarity
    'pearson': scoring.PearsonScoring,      # Pearson correlation coefficient
    'spearman': scoring.SpearmanScoring,    # Spearman rank correlation
    'kendall': scoring.KendallTauScoring,   # Kendall Tau rank correlation
    'ncc': scoring.NCCScoring               # Normalized Cross-Correlation
}

# A registry of supported signal transformation operations.
# Keys are human-readable names (strings) for the transformations,
# and values are the corresponding transformer class constructors.
# This allows for building flexible signal processing pipelines.
SUPPORTED_TRANSFORMER = {
    'mixdown': transform.Mixdown,          # Mixes multi-channel audio to mono
    'normalize': transform.Normalize,      # Normalizes signal amplitude
    'pad': transform.PadZero,              # Pads signal with zeros
    'fft': transform.FFT,                  # Fast Fourier Transform
    'magnitude': transform.Magnitude,      # Computes magnitude of complex numbers (e.g., FFT output)
    'lowpass': transform.LowPass           # Applies a low-pass filter
}
