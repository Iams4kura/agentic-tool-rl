"""Action-level reinforcement learning for structured tool-calling agents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentic-tool-rl")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
