"""Audio Device Manager based on pyAudio."""

from typing import Mapping, Union, List, Optional

import pyaudio

from sonitas.devices import AudioDeviceManager, AudioDevice
from sonitas.pyaudiof import const as paconst


class PyAudioDeviceManager(AudioDeviceManager):
    """Concrete implementation if the AudioDeviceManager using pyAudio."""

    def __init__(self):
        self._pa = pyaudio.PyAudio()

    @classmethod
    def _to_device(cls, index: int, device_info: Mapping[str, Union[str, int, float]]) -> AudioDevice:
        """Parses a device mapping from pyaudio into an `AudioDevice` instance."""
        return AudioDevice(
            index=int(index),
            name=str(device_info.get(paconst.CONST_DEVICE_NAME, 0)),
            default_sample_rate=int(device_info.get(paconst.CONST_DEVICE_DEFAULT_SAMPLE_RATE, 0)),
            max_input_channels=int(device_info.get(paconst.CONST_DEVICE_MAX_INPUT_CHANNELS, 0)),
            max_output_channels=int(device_info.get(paconst.CONST_DEVICE_MAX_OUTPUT_CHANNELS, 0)),
        )

    def select_default_input(self) -> Optional[AudioDevice]:
        devices = self.list(include_output=False)
        if not devices:
            return None
        return devices[0]

    def list(self, include_input: bool = True, include_output: bool = True) -> List[AudioDevice]:
        res = []
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            device = self._to_device(i, dev)
            if include_input and device.max_input_channels > 0:
                res.append(device)
            if include_output and device.max_output_channels > 0:
                res.append(device)

        return res
