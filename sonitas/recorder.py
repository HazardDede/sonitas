"""Recording related base and container classes."""

import wave
from abc import ABCMeta, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from sonitas.devices import AudioDevice


class Recording(BaseModel):
    """
    Container class for a (wave) recording.
    """
    frames: bytes
    channels: int
    sample_size: int
    frame_rate: int

    def to_wav(self, output_file: Path) -> None:
        """
        Writes the frames buffer to the given file location and adds wave related headers.
        """
        out_file = wave.open(str(output_file), 'wb')
        try:
            out_file.setnchannels(self.channels)
            out_file.setsampwidth(self.sample_size)
            out_file.setframerate(self.frame_rate)
            out_file.writeframes(self.frames)
        finally:
            out_file.close()


class Recorder(metaclass=ABCMeta):  # pylint: disable=too-few-public-methods
    """Abstract base class for recording devices."""

    @property
    @abstractmethod
    def current_input_device(self) -> AudioDevice:
        """
        Returns the currently selected audio input device.

        Returns:
            The currently selected audio input device.
        """
        raise NotImplementedError()

    @abstractmethod
    def record(self, duration: int) -> Recording:
        """
        Records audio from the selected input device for a specified duration.

        Args:
            duration: The duration of the recording in seconds.

        Returns:
            A Recording object containing the recorded audio data and metadata.

        Raises:
            InvalidDeviceError: When the used input device is invalid.
            NoInputDeviceError: When no input device is available to record from.
        """
        raise NotImplementedError()
