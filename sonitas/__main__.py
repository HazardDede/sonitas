"""Main entrypoint for running sonitas scripts."""

from sonitas.devices import AudioDevice


class Entrypoint:  # pylint: disable=too-few-public-methods
    """Main entrypoint class."""
    list = AudioDevice.list


if __name__ == '__main__':
    import fire
    fire.Fire(Entrypoint)
