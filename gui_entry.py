"""PyInstaller entrypoint for the GUI build."""

from src import gui


def main() -> None:
    """Launch the GUI."""
    gui.main()


if __name__ == "__main__":
    main()
