"""
Sonitas Command-Line Interface (CLI) Entrypoint.

This module provides the main entry point for interacting with the Sonitas
library via the command line. It uses the `fire` library to automatically
generate a CLI from the `Entrypoint` class and its methods.

This allows users to perform actions such as:
- Comparing two audio files for similarity.
- Listing available audio input and output devices.
- Recording audio from an input device to a WAV file.

To use the CLI, you would typically run this script from your terminal,
e.g., `python -m sonitas <command> [args...]`.
"""
import wave
from pathlib import Path
from typing import Optional, List

from sonitas import const, exc
from sonitas.pyaudiof import PyAudioRecorder, PyAudioDeviceManager
from sonitas.recorder import Recording
from sonitas.similarity import transform, flow, SUPPORTED_SCORING, DEFAULT_SCORING


class Entrypoint:
    """
    Main entry point class for Sonitas CLI commands.

    The methods within this class are exposed as commands by the `fire` library
    when this script is executed.
    """

    @staticmethod
    def compare_files(  # pylint: disable=too-many-branches,broad-exception-caught
            file_path1: str,
            file_path2: str,
            mixdown: bool = True,
            normalize: bool = True,
            pad: bool = True,
            fft: bool = True,
            magnitude: bool = True,
            lowpass: float = 0.1,
            scoring: str = DEFAULT_SCORING,
            verbose: bool = False
    ) -> None:
        """
        Compares two audio WAV files for similarity after applying a series of transformations.

        The comparison involves loading the audio signals, applying selected
        transformations (mixdown, normalization, padding, FFT, magnitude, low-pass filter),
        and then calculating a similarity score using the specified scoring algorithm.

        Args:
            file_path1 (str): Path to the first WAV file.
            file_path2 (str): Path to the second WAV file.
            mixdown (bool): If True, convert signals to mono. Defaults to True.
            normalize (bool): If True, normalize signals (Z-score). Defaults to True.
            pad (bool): If True, pad signals to the same power-of-two length. Defaults to True.
            fft (bool): If True, apply Fast Fourier Transform. Defaults to True.
            magnitude (bool): If True, compute the magnitude of the FFT result. Defaults to True.
            lowpass (float): Keep ratio for the low-pass filter (0.0 to 1.0).
                             A value of 0.0 effectively disables the low-pass filter
                             if it's the only transformation affecting frequency content.
                             If 0, the lowpass step is skipped. Defaults to 0.1.
            scoring (str): The scoring algorithm to use (e.g., 'cosine', 'pearson').
                           Defaults to the `DEFAULT_SCORING` value.
            verbose (bool): If True, print details about the transformers and scorer used.
                            Defaults to False.

        Raises:
            FileNotFoundError: If either `file_path1` or `file_path2` does not exist.
            wave.Error: If either file is not a valid WAV file.
            ValueError: If an invalid `scoring` method is provided or if transformation
                        parameters are invalid (e.g., `lowpass` out of range).
        """
        if scoring not in SUPPORTED_SCORING:
            print(f"Scoring '{scoring}' not supported. Available: {list(SUPPORTED_SCORING.keys())}")
            return

        try:
            signal_a = Recording.from_wav(Path(file_path1)).signal()
            signal_b = Recording.from_wav(Path(file_path2)).signal()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        except wave.Error as e:
            print(f"Error processing WAV file: {e}")
            return

        transformer_steps: List[transform.Transformer] = []
        if mixdown:
            transformer_steps.append(transform.Mixdown())
        if normalize:
            transformer_steps.append(transform.Normalize())
        if pad:
            transformer_steps.append(transform.PadZero())
        if fft:
            transformer_steps.append(transform.FFT())
        if magnitude:
            transformer_steps.append(transform.Magnitude())
        if lowpass > 0.0:  # Only add lowpass if keep_ratio is meaningful
            try:
                transformer_steps.append(transform.LowPass(keep_ratio=lowpass))
            except ValueError as e:  # Catch potential error from LowPass init
                print(f"Error initializing LowPass filter: {e}")
                return
        else:
            if verbose and lowpass != 0.0:  # if lowpass was set to a non-zero invalid value that became 0
                print(f"Info: Lowpass filter skipped due to keep_ratio <= 0.0 (value: {lowpass}).")

        scorer = SUPPORTED_SCORING[scoring]()  # type: ignore

        if verbose:
            print("Transformer Steps:")
            if transformer_steps:
                for tr_step in transformer_steps:
                    print(f'* {tr_step.__class__.__name__}')
            else:
                print("* No transformations applied.")
            print(f"Scoring: {scorer.__class__.__name__}")

        comparison_flow = flow.Flow(transformer_steps, scorer)
        score = comparison_flow.run(signal_a, signal_b)
        print("Similarity Score:", score)

    @staticmethod
    def list(include_input: bool = True, include_output: bool = True) -> None:
        """
        Lists all available audio input and/or output devices.

        This method queries the system for audio devices using PyAudioDeviceManager
        and prints them to the console.

        Args:
            include_input (bool): If True, include input devices in the list.
                                  Defaults to True.
            include_output (bool): If True, include output devices in the list.
                                   Defaults to True.
        """
        manager = PyAudioDeviceManager()
        devices = manager.list(include_input=include_input, include_output=include_output)
        if devices:
            print("Available audio devices:")
            for device in devices:
                print(device)
        else:
            print("No audio devices found matching the criteria.")

    @staticmethod
    def record(
            file_path: str,
            device_index: Optional[int] = None,
            duration: int = const.CONST_DEFAULT_RECORD_DURATION
    ) -> None:
        """
        Records audio from an input device and saves it to a WAV file.

        The user is prompted to press a key to start the recording. If no device
        index is specified, the system's default input device is used.

        Args:
            file_path (str): The path where the output WAV file will be saved.
            device_index (Optional[int]): The index of the recording device to use.
                                          If None, the default input device is selected.
                                          Defaults to None.
            duration (int): The duration of the recording in seconds.
                            Defaults to `sonitas.const.CONST_DEFAULT_RECORD_DURATION` (3 seconds).
        """
        output_path = Path(file_path)
        try:
            recorder = PyAudioRecorder(device_index=device_index)
            print(f"Input device selected: {recorder.current_input_device}")
            input(f"Press Enter to start recording for {duration} seconds...")
            print("Recording...")
            recording = recorder.record(duration)
            recording.to_wav(output_path)
            print(f"Recording complete. Wave file written to {output_path} @ {recording.frame_rate} Hz")
        except (exc.InvalidDeviceError, exc.NoInputDeviceError) as derr:
            print(f"Error: {derr}")
            print("\nPlease check your audio device configuration.")
            print("Available input devices:")
            available_devices = PyAudioDeviceManager().list(include_output=False)
            if not available_devices:
                print("No input devices found.")
        except KeyboardInterrupt:
            print("\nRecording cancelled by user.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"An unexpected error occurred during recording: {e}")


if __name__ == '__main__':
    import fire
    # Expose the Entrypoint class to the command line using python-fire
    # This allows calling methods like:
    # python -m sonitas compare_files file1.wav file2.wav --scoring=pearson
    # python -m sonitas list
    # python -m sonitas record output.wav --duration=5
    fire.Fire(Entrypoint)
