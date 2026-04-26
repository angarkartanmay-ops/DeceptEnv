"""Top-level entry point so HF Spaces can find an `app` ASGI object."""
from server.app import app  # noqa: F401


if __name__ == "__main__":
    from server.app import _run
    _run()
