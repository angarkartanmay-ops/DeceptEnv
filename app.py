"""Hugging Face Spaces entry point.

Some Spaces SDKs prefer a top-level `app.py` that exposes an `app` ASGI object;
we re-export the FastAPI instance from the server package so both work.
"""
from server.app import app  # noqa: F401  (re-exported for HF Spaces)


if __name__ == "__main__":
    from server.app import _run
    _run()
