"""
Audio Device Manager based on PyAudio.

This module provides a concrete implementation of the `AudioDeviceManager`
interface using the PyAudio library. It allows for listing available audio
devices and selecting a default input device by interacting with PyAudio's
APIs.
"""

from typing import Mapping, Union, List, Optional

import pyaudio

from sonitas.devices import AudioDeviceManager, AudioDevice
from sonitas.pyaudiof import const as paconst


class PyAudioDeviceManager(AudioDeviceManager):
    """
    Concrete implementation of the `AudioDeviceManager` using PyAudio.

    This class handles the enumeration and selection of audio devices
    by wrapping PyAudio functionalities. It's responsible for translating
    PyAudio's device information into the `AudioDevice` model used
    within the Sonitas library.
    """

    def __init__(self):
        """
        Initializes the PyAudioDeviceManager.

        This creates an instance of `pyaudio.PyAudio`, which is the main
        entry point for using PyAudio's functionalities. This instance
        will be used for all subsequent PyAudio calls.
        """
        self._pa = pyaudio.PyAudio()

    def __del__(self):
        """
        Cleans up the PyAudio instance when the manager is deleted.

        It's important to terminate the PyAudio instance to release
        system resources.
        """
        if hasattr(self, '_pa') and self._pa:
            self._pa.terminate()

    @classmethod
    def _to_device(cls, index: int, device_info: Mapping[str, Union[str, int, float]]) -> AudioDevice:
        """
        Parses a device information mapping from PyAudio into an `AudioDevice` instance.

        This helper method converts the raw dictionary returned by PyAudio for a
        device into the structured `AudioDevice` Pydantic model.

        Args:
            index: The numerical index of the device as provided by PyAudio.
            device_info: A dictionary containing device properties, obtained
                         from `pyaudio.PyAudio().get_device_info_by_index()`.

        Returns:
            An `AudioDevice` instance populated with information from `device_info`.
        """
        return AudioDevice(
            index=int(index),
            name=str(device_info.get(paconst.CONST_DEVICE_NAME, 0)),
            default_sample_rate=int(device_info.get(paconst.CONST_DEVICE_DEFAULT_SAMPLE_RATE, 0)),
            max_input_channels=int(device_info.get(paconst.CONST_DEVICE_MAX_INPUT_CHANNELS, 0)),
            max_output_channels=int(device_info.get(paconst.CONST_DEVICE_MAX_OUTPUT_CHANNELS, 0)),
        )

    def select_default_input(self) -> Optional[AudioDevice]:
        """
        Selects and returns the system's default audio input device as reported by PyAudio.

        This method attempts to get the default input device directly from PyAudio.
        If no default input device is found or an error occurs, it returns None.

        Returns:
            An `AudioDevice` instance representing the default input device,
            or `None` if no default input device is found or an error occurs.
        """
        devices = self.list(include_output=False)
        if not devices:
            return None
        return devices[0]

    def list(self, include_input: bool = True, include_output: bool = True) -> List[AudioDevice]:
        """
        Lists all available audio devices, with options to filter by type (input/output).

        It iterates through all devices reported by PyAudio, converts them to
        `AudioDevice` objects, and filters them based on their capabilities
        and the provided flags.

        Args:
            include_input: If True, input-capable devices will be included.
            include_output: If True, output-capable devices will be included.

        Returns:
            A list of `AudioDevice` instances matching the criteria.
            An empty list is returned if no devices are found or if both
            `include_input` and `include_output` are False (though the latter
            case would mean no devices match the filter).
        """
        res = []
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            device = self._to_device(i, dev)
            if include_input and device.max_input_channels > 0:
                res.append(device)
            if include_output and device.max_output_channels > 0:
                res.append(device)

        return res
