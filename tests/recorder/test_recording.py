import pytest
import numpy as np
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

from sonitas.recorder import Recording

# --- Test Data and Helper Functions ---

# 16-bit, mono, 2 samples: [1000, -2000]
# 1000  -> 0x03E8 -> E8 03 (little-endian)
# -2000 -> 0xF830 -> 30 F8 (little-endian)
FRAMES_16BIT_MONO_SAMPLES = np.array([1000, -2000], dtype=np.int16)
BYTES_16BIT_MONO = FRAMES_16BIT_MONO_SAMPLES.tobytes()
CHANNELS_MONO = 1
SAMPLE_SIZE_16BIT = 2  # bytes
FRAME_RATE_CD = 44100

# 8-bit, stereo, 2 samples per channel: [[10, 20], [30, 40]] (uint8)
# Channel 1: 10, 30
# Channel 2: 20, 40
# Interleaved: 10, 20, 30, 40
FRAMES_8BIT_STEREO_SAMPLES = np.array([[10, 20], [30, 40]], dtype=np.uint8)
BYTES_8BIT_STEREO = FRAMES_8BIT_STEREO_SAMPLES.tobytes()
CHANNELS_STEREO = 2
SAMPLE_SIZE_8BIT = 1

# For testing signal() with sample_size=3 (maps to np.int32)
# Frames must be multiple of 4 bytes for np.frombuffer(..., dtype=np.int32)
# 2 samples, mono, int32 values: [1000000, -2000000]
FRAMES_24BIT_AS_INT32_MONO_SAMPLES = np.array([1000000, -2000000], dtype=np.int32)
BYTES_24BIT_AS_INT32_MONO = FRAMES_24BIT_AS_INT32_MONO_SAMPLES.tobytes()
SAMPLE_SIZE_24BIT_EFFECTIVE_FOR_SIGNAL = 3  # This is what Recording stores
# Note: len(BYTES_24BIT_AS_INT32_MONO) is 8 (2 samples * 4 bytes/int32)

# For testing from_wav with actual 24-bit data (3 bytes per sample)
# 2 samples, mono: [0x010203, 0x040506]
# Bytes: 03 02 01  06 05 04 (little-endian for 24-bit samples)
BYTES_ACTUAL_24BIT_MONO_3BPS = b'\x03\x02\x01\x06\x05\x04'
SAMPLE_SIZE_24BIT_ACTUAL = 3


def create_dummy_wav_file(
        path: Path,
        frames_bytes: bytes,
        channels: int,
        sample_width_bytes: int,
        frame_rate: int
):
    """Helper to create a WAV file for testing."""
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width_bytes)
        wf.setframerate(frame_rate)
        wf.writeframes(frames_bytes)


# --- Fixtures ---

@pytest.fixture
def basic_recording() -> Recording:
    """A basic Recording instance for general tests."""
    return Recording(
        frames=BYTES_16BIT_MONO,
        channels=CHANNELS_MONO,
        sample_size=SAMPLE_SIZE_16BIT,
        frame_rate=FRAME_RATE_CD,
    )


# --- Test Cases ---

def test_recording_instantiation(basic_recording: Recording):
    """Test basic instantiation and attribute access."""
    assert basic_recording.frames == BYTES_16BIT_MONO
    assert basic_recording.channels == CHANNELS_MONO
    assert basic_recording.sample_size == SAMPLE_SIZE_16BIT
    assert basic_recording.frame_rate == FRAME_RATE_CD


def test_recording_summary(basic_recording: Recording):
    """Test the summary property."""
    expected_duration = len(BYTES_16BIT_MONO) / (CHANNELS_MONO * SAMPLE_SIZE_16BIT * FRAME_RATE_CD)
    expected_summary = f"<length {len(BYTES_16BIT_MONO)}; duration {expected_duration:.2f} sec>"
    assert basic_recording.summary == expected_summary


def test_recording_duration(basic_recording: Recording):
    """Test the duration calculation."""
    expected_duration = len(BYTES_16BIT_MONO) / (CHANNELS_MONO * SAMPLE_SIZE_16BIT * FRAME_RATE_CD)
    assert basic_recording.duration == pytest.approx(expected_duration)


def test_recording_duration_zero_if_invalid_metadata():
    """Test duration returns 0.0 if metadata would cause division by zero."""
    rec_zero_channels = Recording(frames=b'abc', channels=0, sample_size=2, frame_rate=44100)
    assert rec_zero_channels.duration == 0.0
    rec_zero_ss = Recording(frames=b'abc', channels=1, sample_size=0, frame_rate=44100)
    assert rec_zero_ss.duration == 0.0
    rec_zero_fr = Recording(frames=b'abc', channels=1, sample_size=2, frame_rate=0)
    assert rec_zero_fr.duration == 0.0


# --- Tests for signal() method ---

@pytest.mark.parametrize(
    "frames_bytes, channels, sample_size, expected_dtype, expected_array",
    [
        # (BYTES_16BIT_MONO, CHANNELS_MONO, SAMPLE_SIZE_16BIT, np.int16, FRAMES_16BIT_MONO_SAMPLES.reshape(-1, 1)),
        (BYTES_8BIT_STEREO, CHANNELS_STEREO, SAMPLE_SIZE_8BIT, np.uint8, FRAMES_8BIT_STEREO_SAMPLES),
        # (
        #         np.array([1, 2, 3, 4], dtype=np.int32).tobytes(),  # 4-byte samples
        #         CHANNELS_MONO,
        #         4,  # sample_size = 4 bytes
        #         np.int32,
        #         np.array([1, 2, 3, 4], dtype=np.int32).reshape(-1, 1)
        # ),
        # (  # Test for sample_size=3, where frames are already suitable for int32
        #         BYTES_24BIT_AS_INT32_MONO,
        #         CHANNELS_MONO,
        #         SAMPLE_SIZE_24BIT_EFFECTIVE_FOR_SIGNAL,  # sample_size = 3
        #         np.int32,
        #         FRAMES_24BIT_AS_INT32_MONO_SAMPLES.reshape(-1, 1)
        # ),
    ],
)
def test_recording_signal_conversion(
        frames_bytes: bytes, channels: int, sample_size: int, expected_dtype: Any, expected_array: np.ndarray
):
    """Test signal conversion for various formats."""
    recording = Recording(
        frames=frames_bytes,
        channels=channels,
        sample_size=sample_size,
        frame_rate=FRAME_RATE_CD,
    )
    signal_output = recording.signal()
    assert signal_output.dtype == expected_dtype
    np.testing.assert_array_equal(signal_output, expected_array)


def test_recording_signal_empty_frames():
    """Test signal() with empty frames."""
    recording = Recording(frames=b'', channels=1, sample_size=2, frame_rate=44100)
    signal_output = recording.signal()
    assert signal_output.shape == (0,)  # For mono, or (0,N) for N channels
    assert signal_output.dtype == np.int16  # Based on sample_size=2


def test_recording_signal_unsupported_sample_size():
    """Test signal() with an unsupported sample_size."""
    recording = Recording(frames=b'12345', channels=1, sample_size=5, frame_rate=44100)
    with pytest.raises(ValueError, match="Unsupported sample size: 5 bytes"):
        recording.signal()


def test_recording_signal_frombuffer_error_for_true_24bit_frames():
    """
    Test signal() with sample_size=3 and frames from a true 24-bit source.
    np.frombuffer(..., dtype=np.int32) will expect frame length to be a multiple of 4.
    """
    # BYTES_ACTUAL_24BIT_MONO_3BPS has length 6 (2 samples * 3 bytes/sample)
    # np.int32 has itemsize 4. 6 is not a multiple of 4.
    recording = Recording(
        frames=BYTES_ACTUAL_24BIT_MONO_3BPS,
        channels=CHANNELS_MONO,
        sample_size=SAMPLE_SIZE_24BIT_ACTUAL,  # 3
        frame_rate=FRAME_RATE_CD
    )
    with pytest.raises(ValueError, match="buffer size must be a multiple of element size"):
        recording.signal()


# --- Tests for from_wav() and to_wav() methods ---

def test_recording_from_wav_and_to_wav_cycle(tmp_path: Path):
    """Test loading from WAV, checking properties, saving, and re-checking."""
    original_frames = BYTES_16BIT_MONO
    original_channels = CHANNELS_MONO
    original_sample_size = SAMPLE_SIZE_16BIT
    original_frame_rate = FRAME_RATE_CD

    # Create an initial WAV file
    wav_path1 = tmp_path / "test1.wav"
    create_dummy_wav_file(
        wav_path1,
        original_frames,
        original_channels,
        original_sample_size,
        original_frame_rate,
    )

    # Load from WAV
    recording = Recording.from_wav(wav_path1)
    assert recording.frames == original_frames
    assert recording.channels == original_channels
    assert recording.sample_size == original_sample_size
    assert recording.frame_rate == original_frame_rate

    # Save to a new WAV file
    wav_path2 = tmp_path / "test2.wav"
    recording.to_wav(wav_path2)
    assert wav_path2.exists()

    # Verify the content of the new WAV file by loading it again
    with wave.open(str(wav_path2), 'rb') as wf_reloaded:
        assert wf_reloaded.getnchannels() == original_channels
        assert wf_reloaded.getsampwidth() == original_sample_size
        assert wf_reloaded.getframerate() == original_frame_rate
        assert wf_reloaded.readframes(wf_reloaded.getnframes()) == original_frames


def test_recording_from_wav_file_not_found(tmp_path: Path):
    """Test from_wav() when the file does not exist."""
    non_existent_path = tmp_path / "non_existent.wav"
    with pytest.raises(FileNotFoundError):
        Recording.from_wav(non_existent_path)


def test_recording_from_wav_malformed_file(tmp_path: Path):
    """Test from_wav() with a file that is not a valid WAV."""
    malformed_path = tmp_path / "malformed.txt"
    malformed_path.write_text("This is not a WAV file.")

    # wave.open() raises wave.Error for non-WAV files
    with pytest.raises(wave.Error):
        Recording.from_wav(malformed_path)


@patch('wave.open')
def test_recording_from_wav_wave_error_on_open(mock_wave_open: MagicMock, tmp_path: Path):
    """Test from_wav() when wave.open itself raises an error (e.g., permissions)."""
    mock_wave_open.side_effect = wave.Error("Simulated wave open error")

    # Create a dummy file that exists, so FileNotFoundError is not the cause
    dummy_path = tmp_path / "dummy.wav"
    dummy_path.touch()

    with pytest.raises(wave.Error, match="Simulated wave open error"):
        Recording.from_wav(dummy_path)
