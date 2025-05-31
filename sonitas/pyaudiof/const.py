"""
PyAudio Specific Constants.

This module defines constants that correspond to the keys used in the
dictionaries returned by PyAudio when querying for audio device information.
Using these constants instead of raw strings can help prevent typos and
improve code readability when interacting with PyAudio's device info structures.
"""

# Key for accessing the maximum number of input channels for a device
# in a PyAudio device information dictionary.
CONST_DEVICE_MAX_INPUT_CHANNELS = 'maxInputChannels'

# Key for accessing the maximum number of output channels for a device
# in a PyAudio device information dictionary.
CONST_DEVICE_MAX_OUTPUT_CHANNELS = 'maxOutputChannels'

# Key for accessing the human-readable name of a device
# in a PyAudio device information dictionary.
CONST_DEVICE_NAME = 'name'

# Key for accessing the default sample rate (in Hz) for a device
# in a PyAudio device information dictionary.
CONST_DEVICE_DEFAULT_SAMPLE_RATE = 'defaultSampleRate'
