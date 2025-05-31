"""Main entrypoint for running sonitas scripts."""
from pathlib import Path
from typing import Optional, List

from sonitas import const, exc
from sonitas.devices import AudioDevice
from sonitas.pyaudiof import PyAudioRecorder, PyAudioDeviceManager


class Entrypoint:  # pylint: disable=too-few-public-methods
    """Main entrypoint class."""

    @staticmethod
    def list(include_input: bool = True, include_output: bool = True) -> List[AudioDevice]:
        """
        Lists all available input and output audio devices.

        Args:
            include_input (bool): Set to True if input devices should be included.
            include_output (bool): Set to True if output devices should be included.

        Returns:
        A list of available devices (based on the filters).
        """
        manager = PyAudioDeviceManager()
        return manager.list(include_input, include_output)

    @staticmethod
    def record(
            file_path: Path, device_index: Optional[int] = None, duration: int = const.CONST_DEFAULT_RECORD_DURATION
    ) -> None:
        """
        Records the given number of seconds from your input device. The recorded audio will be written to the
        specified file. If no device index is given, it will be automatically determined (if possible).

        Args:
            file_path (Path): The path to the output file.
            device_index (Optional[int]): The recording device to use. If not specified it is automatically determined
              (if possible).
            duration (int): The number of seconds to record. Default is 3.
        """
        try:
            recorder = PyAudioRecorder(device_index=device_index)
            print(f"Input device selected: {recorder.current_input_device}")
            input(f"Press any key to start the recording for {duration} seconds.")
            recording = recorder.record(duration)
            recording.to_wav(file_path)
            print(f"Wave file written to {file_path} @ {recording.frame_rate} Hz")
        except (exc.InvalidDeviceError, exc.NoInputDeviceError) as derr:
            print(
                str(derr),
                f"\nThe following devices are available:\n"
                f"{'\n'.join([str(item) for item in Entrypoint.list(include_output=False)])}"
            )
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    import fire
    fire.Fire(Entrypoint)
