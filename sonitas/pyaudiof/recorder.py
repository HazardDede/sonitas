"""Recording based on pyAudio."""

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
    """

    DEFAULT_CHANNELS = 1
    DEFAULT_RATE = 24000
    DEFAULT_CHUNK = 1024

    def __init__(
            self, device_index: Optional[int] = None
    ):
        """
        Initializer.

        Args:
            device_index (int): The index of the device in pyaudio.
        """
        self.device_manager = PyAudioDeviceManager()
        self._device = self._select_device(device_index)

    @property
    def current_input_device(self) -> AudioDevice:
        return self._device

    def _select_device(self, index: Optional[int] = None) -> AudioDevice:
        assert self.device_manager

        if index is None:
            device = self.device_manager.select_default_input()
            if not device:
                raise exc.NoInputDeviceError("No available input device found.")
            return device

        devices = self.device_manager.list(include_output=False)
        device = next((d for d in devices if d.index == index), None)
        if not device:
            raise exc.InvalidDeviceError(f"Invalid device index: {index}.")

        if device.max_input_channels <= 0:
            raise exc.InvalidDeviceError(f"Device @ index {index} is not an input device.")

        return device

    def record(self, duration: int) -> Recording:
        channels = self._device.max_input_channels
        rate = self._device.default_sample_rate
        chunk_size = self.DEFAULT_CHUNK

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
