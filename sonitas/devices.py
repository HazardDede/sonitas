"""Audio Device related stuff."""
from typing import Mapping, List, Union

import pyaudio
from pydantic import BaseModel


CONST_DEVICE_MAX_INPUT_CHANNELS = 'maxInputChannels'
CONST_DEVICE_MAX_OUTPUT_CHANNELS = 'maxOutputChannels'
CONST_DEVICE_NAME = 'name'
CONST_DEVICE_DEFAULT_SAMPLE_RATE = 'defaultSampleRate'


class AudioDevice(BaseModel):
    """Container class that represents an audio device."""
    index: int
    name: str
    default_sample_rate: int
    max_input_channels: int
    max_output_channels: int

    @classmethod
    def from_pyaudio(cls, index: int, device_info: Mapping[str, Union[str, int, float]]) -> 'AudioDevice':
        """Parses a device mapping from pyaudio into an `AudioDevice` instance."""
        return cls(
            index=int(index),
            name=str(device_info.get(CONST_DEVICE_NAME, 0)),
            default_sample_rate=int(device_info.get(CONST_DEVICE_DEFAULT_SAMPLE_RATE, 0)),
            max_input_channels=int(device_info.get(CONST_DEVICE_MAX_INPUT_CHANNELS, 0)),
            max_output_channels=int(device_info.get(CONST_DEVICE_MAX_OUTPUT_CHANNELS, 0)),
        )

    @classmethod
    def list(cls, finput: bool = True, foutput: bool = True) -> List['AudioDevice']:
        """
        Lists all available input and output devices using pyAudio.

        Args:
            finput (bool): Set to True if input devices should be considered. Defaults to True.
            foutput (bool): Set to True if output devices should be considered. Defaults to True.

        Returns:
            A List of available devices.

        Remarks:
            Please note if a device is considered input and output it will be part of the results even if only
            one filter is enabled.
        """
        pa = pyaudio.PyAudio()
        res = []
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            device = AudioDevice.from_pyaudio(i, dev)
            if finput and device.max_input_channels > 0:
                res.append(device)
            if foutput and device.max_output_channels > 0:
                res.append(device)

        return res

    def __str__(self):
        return (
            f"Index {self.index}: {self.name} "
            f"(Max Input Channels {self.max_input_channels}, "
            f"Max Output Channels {self.max_output_channels}, "
            f"Default @ {self.default_sample_rate} Hz)"
        )
