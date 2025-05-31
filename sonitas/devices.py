"""
Audio Device Representation and Management.

This module provides classes for representing audio devices and an abstract
interface for managing them. It's designed to allow different audio backends
(like PyAudio, ALSA, etc.) to be used interchangeably by providing a common
way to list and select audio devices.

Key Components:
- `AudioDevice`: A Pydantic BaseModel that encapsulates information about a
  single audio device, such as its index, name, sample rate, and channel capabilities.
- `AudioDeviceManager`: An abstract base class defining the essential methods
  that any audio device management implementation should provide, such as
  listing available devices and selecting a default input device.
"""
from abc import ABCMeta, abstractmethod
from typing import List, Optional

from pydantic import BaseModel


class AudioDevice(BaseModel):
    """
    Represents a single audio input or output device.

    This class acts as a data container for device-specific information,
    making it easier to pass around and inspect device properties.
    The `__str__` method provides a human-readable summary of the device.

    Attributes:
        index (int): The system-specific index of the audio device.
        name (str): The human-readable name of the audio device.
        default_sample_rate (int): The default sample rate supported by the
                                   device, in Hz.
        max_input_channels (int): The maximum number of input channels supported
                                  by this device. Will be 0 if it's not an input device.
        max_output_channels (int): The maximum number of output channels supported
                                   by this device. Will be 0 if it's not an output device.
    """
    index: int
    name: str
    default_sample_rate: int
    max_input_channels: int
    max_output_channels: int

    def __str__(self) -> str:
        """
        Provides a string representation of the AudioDevice instance.

        Returns:
            A formatted string summarizing the device's properties.
        """
        return (
            f"Index {self.index}: {self.name} "
            f"(Max Input Channels {self.max_input_channels}, "
            f"Max Output Channels {self.max_output_channels}, "
            f"Default @ {self.default_sample_rate} Hz)"
        )


class AudioDeviceManager(metaclass=ABCMeta):
    """
    Abstract base class for managing and querying audio devices.

    This class defines a common interface for interacting with different
    audio system backends. Concrete implementations (e.g., for PyAudio,
    CoreAudio, ALSA) should inherit from this class and implement its
    abstract methods to provide specific functionality for listing devices
    and selecting default devices.
    """

    @abstractmethod
    def select_default_input(self) -> Optional[AudioDevice]:
        """
        Selects and returns the system's default audio input device.

        Implementations should query the underlying audio system to determine
        the default input device.

        Returns:
            An `AudioDevice` instance representing the default input device
            if one is available and configured; otherwise, `None`.
        """
        raise NotImplementedError("Subclasses must implement select_default_input.")

    @abstractmethod
    def list(self, include_input: bool = True, include_output: bool = True) -> List[AudioDevice]:
        """
        Lists all available audio devices, with options to filter by type.

        Implementations should query the underlying audio system for all
        devices and then filter them based on the `include_input` and
        `include_output` flags.

        Args:
            include_input (bool): If True, input-capable devices will be
                                  included in the returned list. Defaults to True.
            include_output (bool): If True, output-capable devices will be
                                   included in the returned list. Defaults to True.

        Returns:
            A list of `AudioDevice` instances that match the specified criteria.
            Returns an empty list if no devices are found or if both filters
            are False.
        """
        raise NotImplementedError("Subclasses must implement list.")
