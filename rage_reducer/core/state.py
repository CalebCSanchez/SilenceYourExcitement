"""
State management for audio volume monitoring and rage level system.
"""

import logging
import time
from enum import Enum
from typing import Optional, Callable, Deque
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


class VolumeState(Enum):
    """Represents the current volume state based on audio input."""
    CALM = "calm"
    WARNING = "warning" 
    ANGRY = "angry"


class RageLevel(Enum):
    """Represents rage intensity levels based on rage meter fill."""
    CALM = "calm"          # 0.0 fill
    MEDIUM = "medium"      # 0.5+ fill
    ANGRY = "angry"        # 1.0+ fill  
    BLOCKING = "blocking"  # 3.0 fill (full)


@dataclass
class VolumeThresholds:
    """Container for volume threshold values."""
    baseline: float = -30.0  # dBFS
    warning_threshold: float = -27.0  # baseline + 3dB
    angry_threshold: float = -24.0  # baseline + 6dB
    
    def from_baseline(self, baseline_db: float) -> None:
        """Set thresholds based on calibrated baseline."""
        self.baseline = baseline_db
        self.warning_threshold = baseline_db + 3.0
        self.angry_threshold = baseline_db + 6.0
        logger.info(f"Updated thresholds - Baseline: {self.baseline:.1f}dB, "
                   f"Warning: {self.warning_threshold:.1f}dB, "
                   f"Angry: {self.angry_threshold:.1f}dB")


@dataclass
class AudioState:
    """Manages audio state transitions with rage meter system."""
    
    # Thresholds
    thresholds: VolumeThresholds = field(default_factory=VolumeThresholds)
    
    # Current state
    current_state: VolumeState = VolumeState.CALM
    previous_state: VolumeState = VolumeState.CALM
    
    # Rage meter system (0.0 to 3.0)
    rage_level: float = 0.0
    max_rage_level: float = 3.0
    current_rage_level: RageLevel = RageLevel.CALM
    previous_rage_level: RageLevel = RageLevel.CALM
    
    # Rage timing constants  
    rage_fill_rate: float = 1.0  # units per second when angry
    rage_decay_rate: float = 0.25  # units per second when calm
    calm_stagnant_duration: float = 1.0  # seconds to wait before decay
    
    # Rage timing state
    last_rage_update: float = field(default_factory=time.time)
    calm_start_time: Optional[float] = None
    
    # Debouncing
    debounce_frames: int = 3  # Require 3 consecutive frames for ANGRY state
    state_history: Deque[VolumeState] = field(default_factory=lambda: deque(maxlen=5))
    
    # Timing
    last_state_change: float = field(default_factory=time.time)
    angry_start_time: Optional[float] = None
    
    # Callbacks
    on_state_change: Optional[Callable[[VolumeState, VolumeState], None]] = None
    on_rage_change: Optional[Callable[[RageLevel, RageLevel, float], None]] = None
    
    def __post_init__(self):
        """Initialize state history."""
        self.state_history.extend([VolumeState.CALM] * self.debounce_frames)
    
    def update_volume(self, volume_db: float, has_voice: bool = True) -> VolumeState:
        """
        Update state based on current volume level.
        
        Args:
            volume_db: Current volume in dBFS
            has_voice: Whether voice activity was detected
            
        Returns:
            Current volume state after update
        """
        current_time = time.time()
        
        if not has_voice:
            # No voice detected, transition to calm
            new_state = self._transition_to_calm()
        else:
            # Determine raw state based on volume
            if volume_db >= self.thresholds.angry_threshold:
                raw_state = VolumeState.ANGRY
            elif volume_db >= self.thresholds.warning_threshold:
                raw_state = VolumeState.WARNING
            else:
                raw_state = VolumeState.CALM
            
            # Add to history
            self.state_history.append(raw_state)
            
            # Apply debouncing logic
            new_state = self._apply_debouncing(raw_state)
            
            # Update state if changed
            if new_state != self.current_state:
                self._change_state(new_state)
        
        # Update rage meter based on current state
        self._update_rage_meter(current_time)
        
        return self.current_state
    
    def _update_rage_meter(self, current_time: float) -> None:
        """Update rage meter based on current state and timing."""
        time_delta = current_time - self.last_rage_update
        self.last_rage_update = current_time
        
        # Cap time delta to prevent huge jumps
        time_delta = min(time_delta, 0.5)
        
        if self.current_state == VolumeState.ANGRY:
            # Fill rage meter at full speed when angry
            self.rage_level = min(self.max_rage_level, self.rage_level + (self.rage_fill_rate * time_delta))
            self.calm_start_time = None
            
        elif self.current_state == VolumeState.WARNING:
            # Fill rage meter at half speed when warning (building up tension)
            warning_fill_rate = self.rage_fill_rate * 0.5  # 0.5 units per second
            self.rage_level = min(self.max_rage_level, self.rage_level + (warning_fill_rate * time_delta))
            self.calm_start_time = None
            
        elif self.current_state == VolumeState.CALM:
            # Handle calm state - wait for stagnant period, then decay
            if self.rage_level > 0.0:  # Only process if there's rage to decay
                if self.calm_start_time is None:
                    self.calm_start_time = current_time
                
                # Check if stagnant period has passed
                time_since_calm = current_time - self.calm_start_time
                if time_since_calm >= self.calm_stagnant_duration:
                    # Start decaying rage level
                    self.rage_level = max(0.0, self.rage_level - (self.rage_decay_rate * time_delta))
        
        # Determine rage level category
        new_rage_level = self._get_rage_level_from_fill()
        
        # Check if rage level changed - add minimum change threshold to prevent flashing
        rage_level_changed = new_rage_level != self.current_rage_level
        
        # Only notify on significant changes or high rage levels
        # Add debouncing: don't notify too frequently for the same level
        should_notify = False
        if rage_level_changed:
            should_notify = True
        elif self.rage_level >= 2.0:
            # For high rage levels, only log every 0.5 seconds to reduce spam
            if not hasattr(self, '_last_high_rage_log'):
                self._last_high_rage_log = 0
            if current_time - self._last_high_rage_log >= 0.5:
                should_notify = True
                self._last_high_rage_log = current_time
        
        if should_notify:
            if rage_level_changed:
                self.previous_rage_level = self.current_rage_level
                self.current_rage_level = new_rage_level
                logger.info(f"Rage level changed: {self.previous_rage_level.value} -> {new_rage_level.value} (fill: {self.rage_level:.2f})")
                
                # Call the callback with the correct rage level change
                if self.on_rage_change:
                    self.on_rage_change(self.previous_rage_level, self.current_rage_level, self.rage_level)
            else:
                logger.info(f"High rage level: {new_rage_level.value} (fill: {self.rage_level:.2f})")
    
    def _get_rage_level_from_fill(self) -> RageLevel:
        """Get rage level category from current fill amount."""
        if self.rage_level >= 3.0:
            return RageLevel.BLOCKING
        elif self.rage_level >= 1.0:
            return RageLevel.ANGRY
        elif self.rage_level >= 0.5:
            return RageLevel.MEDIUM
        else:
            return RageLevel.CALM
    
    def get_rage_fill_percentage(self) -> float:
        """Get rage meter fill as percentage (0.0 to 1.0)."""
        return self.rage_level / self.max_rage_level
    
    def get_rage_opacity(self) -> float:
        """Get overlay opacity based on rage level (0.0 to 1.0)."""
        if self.current_rage_level == RageLevel.BLOCKING:
            return 1.0  # Full red screen
        elif self.current_rage_level == RageLevel.ANGRY:
            # Scale from 0.6 to 0.95 between fill 1.0 and 3.0
            progress = (self.rage_level - 1.0) / 2.0  # 0.0 to 1.0
            return 0.6 + (progress * 0.35)
        elif self.current_rage_level == RageLevel.MEDIUM:
            # Scale from 0.2 to 0.6 between fill 0.5 and 1.0  
            progress = (self.rage_level - 0.5) / 0.5  # 0.0 to 1.0
            return 0.2 + (progress * 0.4)
        else:
            # Fade out from 0.2 to 0.0 between fill 0.5 and 0.0
            if self.rage_level > 0.0:
                return (self.rage_level / 0.5) * 0.2
            return 0.0
    
    def _apply_debouncing(self, raw_state: VolumeState) -> VolumeState:
        """Apply debouncing logic to prevent rapid state changes."""
        
        # For ANGRY state, require consecutive frames
        if raw_state == VolumeState.ANGRY:
            angry_count = sum(1 for s in self.state_history if s == VolumeState.ANGRY)
            if angry_count >= self.debounce_frames:
                return VolumeState.ANGRY
            else:
                # Not enough consecutive angry frames, stay in current state
                return self.current_state
        
        # For WARNING state, no debouncing needed
        elif raw_state == VolumeState.WARNING:
            return VolumeState.WARNING
        
        # For CALM state, immediate transition
        else:
            return VolumeState.CALM
    
    def _transition_to_calm(self) -> VolumeState:
        """Transition back to calm state."""
        if self.current_state != VolumeState.CALM:
            self._change_state(VolumeState.CALM)
        return self.current_state
    
    def _change_state(self, new_state: VolumeState) -> None:
        """Handle state transition."""
        self.previous_state = self.current_state
        self.current_state = new_state
        self.last_state_change = time.time()
        
        # Track angry state duration
        if new_state == VolumeState.ANGRY and self.previous_state != VolumeState.ANGRY:
            self.angry_start_time = time.time()
        elif new_state != VolumeState.ANGRY:
            self.angry_start_time = None
        
        logger.debug(f"State changed: {self.previous_state.value} -> {new_state.value}")
        
        # Notify listeners
        if self.on_state_change:
            self.on_state_change(self.previous_state, self.current_state)
    
    def get_angry_duration(self) -> float:
        """Get duration of current angry state in seconds."""
        if self.current_state == VolumeState.ANGRY and self.angry_start_time:
            return time.time() - self.angry_start_time
        return 0.0
    
    def reset(self) -> None:
        """Reset to calm state and clear rage meter."""
        self.current_state = VolumeState.CALM
        self.previous_state = VolumeState.CALM
        self.state_history.clear()
        self.state_history.extend([VolumeState.CALM] * self.debounce_frames)
        self.angry_start_time = None
        
        # Reset rage meter
        self.rage_level = 0.0
        self.current_rage_level = RageLevel.CALM
        self.previous_rage_level = RageLevel.CALM
        self.calm_start_time = None
        self.last_rage_update = time.time()
        
        logger.info("Audio state and rage meter reset to CALM")


def calculate_rms_db(audio_data: np.ndarray) -> float:
    """
    Calculate RMS volume in dBFS.
    
    Args:
        audio_data: Audio samples (normalized to [-1, 1])
        
    Returns:
        RMS volume in dBFS
    """
    if len(audio_data) == 0:
        return -np.inf
    
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms == 0:
        return -np.inf
    
    # Convert to dBFS (0 dBFS = maximum possible level)
    db_fs = 20 * np.log10(rms)
    return db_fs


def calculate_baseline_from_samples(volume_samples: list[float]) -> float:
    """
    Calculate baseline volume from calibration samples.
    
    Args:
        volume_samples: List of volume measurements in dBFS
        
    Returns:
        Baseline volume (median) in dBFS
    """
    if not volume_samples:
        return -30.0  # Default baseline
    
    # Filter out silence (-inf values)
    valid_samples = [v for v in volume_samples if v > -60.0]
    
    if not valid_samples:
        return -30.0
    
    baseline = np.median(valid_samples)
    logger.info(f"Calculated baseline from {len(valid_samples)} samples: {baseline:.1f}dB")
    return baseline 