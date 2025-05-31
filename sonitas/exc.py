"""Exception module."""


class InvalidDeviceError(Exception):
    """Is thrown when a device (or its index) is invalid."""


class NoInputDeviceError(Exception):
    """Is thrown when no input device is available."""
