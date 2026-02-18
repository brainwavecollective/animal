"""Animal - Real-time emotional affect engine for robots."""

from .animal import Animal
from .config import Config
from .utils.time_source import TimeSource

__all__ = ["Animal", "Config", "TimeSource"]
