"""
Recording and Audio Data Handling.

This module provides classes for managing audio recordings and defining
an interface for audio recording devices.

Key Components:
- `Recording`: A Pydantic BaseModel that encapsulates raw audio frames and
  metadata (channels, sample size, frame rate). It includes methods for
  loading from WAV files, calculating duration, converting to a numerical
  signal (NumPy array), and saving back to a WAV file.
- `Recorder`: An abstract base class defining the essential properties and
  methods that any audio recorder implementation should provide, such as
  accessing the current input device and initiating a recording.
"""

import wave
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from pydantic import BaseModel, Field

from sonitas.devices import AudioDevice
from sonitas.typestore import Signal


class Recording(BaseModel):
    """
    Container class for an audio recording, typically from a WAV file or live input.

    This class stores raw audio frames along with essential metadata like the
    number of channels, sample size (bytes per sample), and frame rate (samples
    per second). It provides utilities to load from/save to WAV files,
    calculate duration, and convert the raw frames into a NumPy array
    representation (Signal).

    Attributes:
        frames (bytes): The raw audio data. Not directly shown in `repr` for brevity.
        channels (int): The number of audio channels (e.g., 1 for mono, 2 for stereo).
        sample_size (int): The size of each audio sample in bytes (e.g., 2 for 16-bit audio).
        frame_rate (int): The number of frames per second (e.g., 44100 Hz).
    """
    frames: bytes = Field(repr=False)
    channels: int
    sample_size: int
    frame_rate: int

    @property
    def summary(self) -> str:
        """
        Provides a brief summary string of the recording.

        Returns:
            A string indicating the length of the frames in bytes and the
            duration of the recording in seconds.
        """
        return f"<length {len(self.frames)}; duration {self.duration:.2f} sec>"

    @classmethod
    def from_wav(cls, file_path: Union[str, Path]) -> 'Recording':
        """
        Creates a Recording instance by loading data from a WAV file.

        Args:
            file_path: The path to the WAV file (can be a string or a Path object).

        Returns:
            A new Recording instance populated with data from the WAV file.

        Raises:
            FileNotFoundError: If the specified WAV file does not exist.
            wave.Error: If the file is not a valid WAV file or cannot be opened.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File '{file_path}' not found.")

        with wave.open(str(file_path), 'rb') as wf:
            return cls(
                frames=wf.readframes(wf.getnframes()),
                channels=wf.getnchannels(),
                sample_size=wf.getsampwidth(),
                frame_rate=wf.getframerate()
            )

    @property
    def duration(self) -> float:
        """
        Calculates and returns the duration of the recording in seconds.

        The duration is derived from the total number of bytes in `frames`,
        the number of channels, the sample size (bytes per sample), and the
        frame rate.

        Returns:
            The duration of the recording in seconds.
        """
        if self.channels == 0 or self.sample_size == 0 or self.frame_rate == 0:
            return 0.0  # Avoid division by zero if metadata is invalid

        return len(self.frames) / self.channels / self.sample_size / self.frame_rate

    def signal(self) -> Signal:
        """
        Converts the raw audio frames into a NumPy array (Signal).

        The method interprets the byte string `frames` based on the `sample_size`
        to determine the correct NumPy data type. For multi-channel audio,
        the resulting array is reshaped to have dimensions (n_samples, n_channels).

        Returns:
            A NumPy array representing the audio signal. For mono audio, this is
            a 1D array. For stereo or multi-channel audio, this is a 2D array
            where each row is a sample and each column is a channel.

        Raises:
            ValueError: If the `sample_size` is unsupported or if the frames
                        cannot be correctly reshaped according to the number of channels.
        """
        # Choose correct numpy dtype based on sample width
        dtype_map = {
            1: np.uint8,  # 8-bit PCM (unsigned)
            2: np.int16,  # 16-bit PCM (signed)
            3: np.int32,  # 24-bit PCM stored as 32-bit container
            4: np.int32,  # 32-bit PCM
        }

        dtype = dtype_map.get(self.sample_size)
        if dtype is None:
            raise ValueError(f"Unsupported sample size: {self.sample_size} bytes")

        if len(self.frames) == 0:
            return np.array([], dtype=dtype)  # Return empty array if no frames

        signal = np.frombuffer(self.frames, dtype=dtype)

        # Reshape based on number of channels
        if self.channels > 1:
            signal = signal.reshape(-1, self.channels)

        return signal

    def to_wav(self, output_file: Path) -> None:
        """
        Writes the recording's audio frames to a WAV file.

        The WAV file will be created or overwritten at the specified path
        with the metadata (channels, sample size, frame rate) from this
        Recording instance.

        Args:
            output_file: The path where the WAV file will be saved
                         (can be a string or a Path object).
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
    """
    Abstract base class for audio recording devices.

    This class defines the common interface that all specific recorder
    implementations (e.g., PyAudioRecorder, AlsaRecorder) must adhere to.
    It ensures that different recording backends can be used interchangeably
    within the application.
    """

    @property
    @abstractmethod
    def current_input_device(self) -> AudioDevice:
        """
        Retrieves the currently selected audio input device.

        Implementations should return an `AudioDevice` object representing
        the active device being used for recording.

        Returns:
            The `AudioDevice` object for the current input.
        """
        raise NotImplementedError()

    @abstractmethod
    def record(self, duration: int) -> Recording:
        """
        Records audio from the selected input device for a specified duration.

        Implementations should handle the specifics of interacting with the
        audio hardware or library to capture audio data.

        Args:
            duration: The desired duration of the recording in seconds.

        Returns:
            A `Recording` object containing the captured audio data (frames)
            and its associated metadata (channels, sample rate, sample width).

        Raises:
            exc.InvalidDeviceError: (Example) If the configured input device is invalid or unusable.
            exc.NoInputDeviceError: (Example) If no input device is available or selected.
            # Other device-specific or recording-related errors may be raised.
        """
        raise NotImplementedError()
