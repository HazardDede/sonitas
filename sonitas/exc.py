"""
Sonitas Custom Exception Module.

This module defines custom exception classes used throughout the Sonitas
library. These exceptions provide more specific error information than
built-in exceptions, aiding in debugging and error handling for
applications using Sonitas.
"""


class InvalidDeviceError(Exception):
    """
    Raised when an audio device is invalid or cannot be used.

    This can occur if:
    - An invalid device index is provided.
    - The specified device does not support the required operation (e.g.,
      attempting to use an output-only device as an input).
    - The device is not found or is unavailable.
    """


class NoInputDeviceError(Exception):
    """
    Raised when no suitable audio input device can be found or is available.

    This typically occurs if:
    - There are no audio input devices connected to the system.
    - The default input device cannot be determined or is not functional.
    """
