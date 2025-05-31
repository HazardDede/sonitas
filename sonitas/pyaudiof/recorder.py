"""
PyAudio-based Audio Recorder Implementation.

This module provides the `PyAudioRecorder` class, which uses the PyAudio
library to capture audio from a specified or default input device.
It conforms to the `Recorder` interface defined in `sonitas.recorder`.
"""

from typing import Optional

import pyaudio

from sonitas import exc
from sonitas.devices import AudioDevice
from sonitas.recorder import Recorder, Recording
from .manager import PyAudioDeviceManager


class PyAudioRecorder(Recorder):
    """
    A recorder implementation that uses the PyAudio library to capture audio
    from an input device.

    This class manages the selection of an audio input device and provides
    a method to record audio for a specified duration, returning the
    captured audio as a `Recording` object.
    """

    DEFAULT_CHANNELS = 1
    DEFAULT_RATE = 24000
    DEFAULT_CHUNK = 1024

    def __init__(
            self, device_index: Optional[int] = None
    ):
        """
        Initializes the PyAudioRecorder.

        Sets up the device manager and selects the audio input device to be used
        for recording. If `device_index` is not provided, the default system
        input device will be selected.

        Args:
            device_index (Optional[int]): The index of the PyAudio input device.
                                          If None, the default input device is used.
        """
        self.device_manager = PyAudioDeviceManager()
        self._device = self._select_device(device_index)

    @property
    def current_input_device(self) -> AudioDevice:
        """
        Returns the currently selected audio input device for this recorder.
        """
        return self._device

    def _select_device(self, index: Optional[int] = None) -> AudioDevice:
        """
        Selects an audio input device based on the provided index or system default.

        If an `index` is provided, it attempts to find and use that device.
        If `index` is None, it attempts to use the system's default input device.

        Args:
            index (Optional[int]): The numerical index of the desired input device.

        Returns:
            AudioDevice: The selected audio input device.

        Raises:
            exc.NoInputDeviceError: If no input device can be found (e.g., when
                                    `index` is None and no default is available,
                                    or if the system has no input devices).
            exc.InvalidDeviceError: If the specified `index` is invalid, does not
                                    correspond to an input device, or the device
                                    has no input channels.
        """
        assert self.device_manager

        if index is None:
            device = self.device_manager.select_default_input()
            if not device:
                raise exc.NoInputDeviceError("No available input device found.")
            return device

        devices = self.device_manager.list()
        device = next((d for d in devices if d.index == index), None)
        if not device:
            raise exc.InvalidDeviceError(f"Invalid device index: {index}.")

        if device.max_input_channels <= 0:
            raise exc.InvalidDeviceError(f"Device @ index {index} is not an input device.")

        return device

    def record(self, duration: int) -> Recording:
        """
        Records audio from the selected input device for a specified duration.

        This method opens an audio stream using PyAudio, reads data in chunks,
        and then compiles it into a `Recording` object. Resources (stream and
        PyAudio instance) are cleaned up automatically.

        Args:
            duration (int): The duration of the recording in seconds.

        Returns:
            Recording: An object containing the recorded audio frames, number of
                       channels, sample size (in bytes), and frame rate.

        Raises:
            # PyAudio might raise exceptions during stream opening or reading
            # if there are issues with the device or system audio.
            # e.g., IOError if the device is unavailable or parameters are unsupported.
        """
        channels = self._device.max_input_channels
        rate = self._device.default_sample_rate
        chunk_size = self.DEFAULT_CHUNK

        # Validate that the selected device actually has input channels and a valid sample rate
        if channels <= 0:
            raise exc.InvalidDeviceError(f"Selected device '{self._device.name}' has no input channels.")
        if rate <= 0:
            raise exc.InvalidDeviceError(
                f"Selected device '{self._device.name}' has an invalid default sample rate: {rate} Hz."
            )

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=self._device.index,
            frames_per_buffer=chunk_size
        )

        try:
            frames = []
            for _ in range(0, int(rate / chunk_size * duration)):
                data = stream.read(chunk_size)
                frames.append(data)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

        return Recording(
            frames=b''.join(frames),
            channels=channels,
            sample_size=pa.get_sample_size(pyaudio.paInt16),
            frame_rate=rate
        )
