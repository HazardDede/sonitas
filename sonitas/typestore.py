"""
Sonitas Type Store.

This module serves as a central place to define custom type aliases used
throughout the Sonitas project. This helps in improving code readability
and maintainability by providing meaningful names for common data structures.

Currently, it defines a type alias for signals, which are represented
as NumPy arrays.
"""
import numpy as np

# Represents a signal, typically a time-series or frequency-domain data,
# as a NumPy array. This alias is used for type hinting and to provide
# semantic meaning to n-dimensional arrays used as signals.
Signal = np.ndarray
