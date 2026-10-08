"""PyInstaller entry point; keep imports independent of the legacy application."""

from culling_engine.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
