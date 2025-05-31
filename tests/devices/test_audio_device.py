import pytest

from sonitas.devices import AudioDevice

# Test data for creating AudioDevice instances
VALID_DEVICE_DATA_INPUT_ONLY = {
    "index": 0,
    "name": "Microphone Array (Realtek Audio)",
    "default_sample_rate": 48000,
    "max_input_channels": 2,
    "max_output_channels": 0,
}

VALID_DEVICE_DATA_OUTPUT_ONLY = {
    "index": 1,
    "name": "Speakers (Realtek Audio)",
    "default_sample_rate": 44100,
    "max_input_channels": 0,
    "max_output_channels": 2,
}

VALID_DEVICE_DATA_INPUT_OUTPUT = {
    "index": 2,
    "name": "Headset (Logitech USB)",
    "default_sample_rate": 48000,
    "max_input_channels": 1,
    "max_output_channels": 2,
}


def test_audio_device_creation_input_only():
    """Test creating an AudioDevice instance for an input-only device."""
    device_data = VALID_DEVICE_DATA_INPUT_ONLY
    device = AudioDevice(**device_data)

    assert device.index == device_data["index"]
    assert device.name == device_data["name"]
    assert device.default_sample_rate == device_data["default_sample_rate"]
    assert device.max_input_channels == device_data["max_input_channels"]
    assert device.max_output_channels == device_data["max_output_channels"]


def test_audio_device_creation_output_only():
    """Test creating an AudioDevice instance for an output-only device."""
    device_data = VALID_DEVICE_DATA_OUTPUT_ONLY
    device = AudioDevice(**device_data)

    assert device.index == device_data["index"]
    assert device.name == device_data["name"]
    assert device.default_sample_rate == device_data["default_sample_rate"]
    assert device.max_input_channels == device_data["max_input_channels"]
    assert device.max_output_channels == device_data["max_output_channels"]


def test_audio_device_creation_input_output():
    """Test creating an AudioDevice instance for an input/output device."""
    device_data = VALID_DEVICE_DATA_INPUT_OUTPUT
    device = AudioDevice(**device_data)

    assert device.index == device_data["index"]
    assert device.name == device_data["name"]
    assert device.default_sample_rate == device_data["default_sample_rate"]
    assert device.max_input_channels == device_data["max_input_channels"]
    assert device.max_output_channels == device_data["max_output_channels"]


def test_audio_device_str_representation_input_only():
    """Test the string representation of an input-only AudioDevice."""
    device_data = VALID_DEVICE_DATA_INPUT_ONLY
    device = AudioDevice(**device_data)
    expected_str = (
        f"Index {device_data['index']}: {device_data['name']} "
        f"(Max Input Channels {device_data['max_input_channels']}, "
        f"Max Output Channels {device_data['max_output_channels']}, "
        f"Default @ {device_data['default_sample_rate']} Hz)"
    )
    assert str(device) == expected_str


def test_audio_device_str_representation_output_only():
    """Test the string representation of an output-only AudioDevice."""
    device_data = VALID_DEVICE_DATA_OUTPUT_ONLY
    device = AudioDevice(**device_data)
    expected_str = (
        f"Index {device_data['index']}: {device_data['name']} "
        f"(Max Input Channels {device_data['max_input_channels']}, "
        f"Max Output Channels {device_data['max_output_channels']}, "
        f"Default @ {device_data['default_sample_rate']} Hz)"
    )
    assert str(device) == expected_str


def test_audio_device_str_representation_input_output():
    """Test the string representation of an input/output AudioDevice."""
    device_data = VALID_DEVICE_DATA_INPUT_OUTPUT
    device = AudioDevice(**device_data)
    expected_str = (
        f"Index {device_data['index']}: {device_data['name']} "
        f"(Max Input Channels {device_data['max_input_channels']}, "
        f"Max Output Channels {device_data['max_output_channels']}, "
        f"Default @ {device_data['default_sample_rate']} Hz)"
    )
    assert str(device) == expected_str


def test_audio_device_invalid_type_for_index():
    """Test that Pydantic raises ValidationError for incorrect data types."""
    invalid_data = VALID_DEVICE_DATA_INPUT_ONLY.copy()
    invalid_data["index"] = "not_an_integer"  # Invalid type for index

    with pytest.raises(ValueError):  # Pydantic v1 raises ValidationError, v2 might raise ValueError for coercion
        AudioDevice(**invalid_data)


def test_audio_device_missing_required_field():
    """Test that Pydantic raises ValidationError if a required field is missing."""
    incomplete_data = VALID_DEVICE_DATA_INPUT_ONLY.copy()
    del incomplete_data["name"]  # Missing 'name'

    with pytest.raises(ValueError):  # Pydantic v1 raises ValidationError, v2 might raise ValueError
        AudioDevice(**incomplete_data)
