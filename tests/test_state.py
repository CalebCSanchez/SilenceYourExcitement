"""Tests for state management."""

import pytest
import numpy as np
from rage_reducer.core.state import (
    VolumeState, VolumeThresholds, AudioState,
    calculate_rms_db, calculate_baseline_from_samples
)


class TestVolumeThresholds:
    """Test volume threshold calculations."""
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = VolumeThresholds()
        assert thresholds.baseline == -30.0
        assert thresholds.warning_threshold == -27.0
        assert thresholds.angry_threshold == -24.0
    
    def test_from_baseline(self):
        """Test threshold calculation from baseline."""
        thresholds = VolumeThresholds()
        thresholds.from_baseline(-25.0)
        
        assert thresholds.baseline == -25.0
        assert thresholds.warning_threshold == -22.0
        assert thresholds.angry_threshold == -19.0


class TestAudioState:
    """Test audio state management."""
    
    def test_initial_state(self):
        """Test initial state is calm."""
        state = AudioState()
        assert state.current_state == VolumeState.CALM
        assert state.previous_state == VolumeState.CALM
    
    def test_volume_state_transitions(self):
        """Test state transitions based on volume."""
        state = AudioState()
        
        # Test calm state
        result = state.update_volume(-35.0, has_voice=True)
        assert result == VolumeState.CALM
        
        # Test warning state
        result = state.update_volume(-25.0, has_voice=True)
        assert result == VolumeState.WARNING
        
        # Test angry state (requires debouncing)
        for _ in range(3):
            result = state.update_volume(-20.0, has_voice=True)
        assert result == VolumeState.ANGRY
    
    def test_no_voice_detection(self):
        """Test behavior when no voice is detected."""
        state = AudioState()
        
        # Loud volume but no voice should stay calm
        result = state.update_volume(-15.0, has_voice=False)
        assert result == VolumeState.CALM
    
    def test_debouncing(self):
        """Test angry state debouncing."""
        state = AudioState()
        
        # Single angry frame should not trigger angry state
        result = state.update_volume(-20.0, has_voice=True)
        assert result == VolumeState.CALM
        
        # Multiple consecutive angry frames should trigger
        for _ in range(3):
            result = state.update_volume(-20.0, has_voice=True)
        assert result == VolumeState.ANGRY


class TestVolumeCalculations:
    """Test volume calculation functions."""
    
    def test_calculate_rms_db(self):
        """Test RMS dBFS calculation."""
        # Test silence
        silence = np.zeros(1000)
        result = calculate_rms_db(silence)
        assert result == -np.inf
        
        # Test full scale signal
        full_scale = np.ones(1000)
        result = calculate_rms_db(full_scale)
        assert abs(result - 0.0) < 0.1  # Should be close to 0 dBFS
        
        # Test half scale signal
        half_scale = np.ones(1000) * 0.5
        result = calculate_rms_db(half_scale)
        assert abs(result - (-6.0)) < 0.1  # Should be close to -6 dBFS
    
    def test_calculate_baseline_from_samples(self):
        """Test baseline calculation from samples."""
        # Test empty samples
        result = calculate_baseline_from_samples([])
        assert result == -30.0
        
        # Test with valid samples
        samples = [-25.0, -28.0, -30.0, -26.0, -29.0]
        result = calculate_baseline_from_samples(samples)
        assert result == -28.0  # Median
        
        # Test filtering out silence
        samples_with_silence = [-25.0, -70.0, -28.0, -30.0]
        result = calculate_baseline_from_samples(samples_with_silence)
        assert abs(result - (-27.5)) < 0.1  # Median of valid samples


if __name__ == "__main__":
    pytest.main([__file__]) 