"""
Sonitas PyAudio Functionality Subpackage.

This package provides concrete implementations of Sonitas interfaces using the
PyAudio library. It serves as the bridge between the abstract Sonitas
components (like `Recorder` and `AudioDeviceManager`) and the specific
functionalities offered by PyAudio for audio recording and device management.

This `__init__.py` file makes the primary classes of this subpackage,
`PyAudioRecorder` and `PyAudioDeviceManager`, directly importable from
`sonitas.pyaudiof`.
"""

from .recorder import PyAudioRecorder
from .manager import PyAudioDeviceManager

# The `__all__` list defines the public API of this package.
# When a client executes `from sonitas.pyaudiof import *`, only the names
# listed in `__all__` will be imported. This helps in keeping the namespace
# clean and explicitly states what is intended for external use.
__all__ = [
    'PyAudioRecorder',        # Recorder implementation using PyAudio.
    'PyAudioDeviceManager'    # Audio device manager using PyAudio.
]
