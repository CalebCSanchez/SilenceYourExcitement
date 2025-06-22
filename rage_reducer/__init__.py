"""
Rage-Reducer: A desktop utility to reduce gaming rage through visual feedback.

This package provides real-time audio monitoring with screen overlay effects
to help users manage their voice volume during intense gaming sessions.
"""

__version__ = "1.0.0"
__author__ = "Rage-Reducer Team"

from .core.state import AudioState, VolumeState
from .core.config import Config

__all__ = ["AudioState", "VolumeState", "Config"] 