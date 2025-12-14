"""Reusable code for the MABe mouse behavior pipelines."""

from importlib import metadata


def __getattr__(name: str) -> str:
    if name == "__version__":
        try:
            return metadata.version("mabe")
        except metadata.PackageNotFoundError:
            return "0.0.0"
    raise AttributeError(name)
