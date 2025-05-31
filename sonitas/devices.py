"""Audio Device related stuff."""
from abc import ABCMeta, abstractmethod
from typing import List, Optional

from pydantic import BaseModel


class AudioDevice(BaseModel):
    """Container class that represents an audio device."""
    index: int
    name: str
    default_sample_rate: int
    max_input_channels: int
    max_output_channels: int

    def __str__(self):
        return (
            f"Index {self.index}: {self.name} "
            f"(Max Input Channels {self.max_input_channels}, "
            f"Max Output Channels {self.max_output_channels}, "
            f"Default @ {self.default_sample_rate} Hz)"
        )


class AudioDeviceManager(metaclass=ABCMeta):
    """Abstract base class for managing audio devices."""

    @abstractmethod
    def select_default_input(self) -> Optional[AudioDevice]:
        """
        Selects the default input device.

        Returns:
            The default input device if any is available. Otherwise, None.
        """
        raise NotImplementedError()

    @abstractmethod
    def list(self, include_input: bool = True, include_output: bool = True) -> List[AudioDevice]:
        """
        Lists all available input and output audio devices.

        Args:
            include_input (bool): Set to True if input devices should be included. Defaults to True.
            include_output (bool): Set to True if output devices should be included. Defaults to True.

        Returns:
            A list of available devices (based on the filters).
        """
        raise NotImplementedError()
