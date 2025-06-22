"""
Noise suppression using noisereduce (alternative to RNNoise).
Removes background noise like keyboard clicks, fan noise, etc.
"""

import logging
from typing import Optional

import numpy as np

# Try to import noisereduce as an alternative to RNNoise
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    logging.warning("noisereduce not available - noise suppression disabled")

# Keep the original RNNoise import for potential future use
try:
    import rnnoise
    HAS_RNNOISE = True
except ImportError:
    HAS_RNNOISE = False
    if not HAS_NOISEREDUCE:
        logging.warning("RNNoise not available - noise suppression disabled")

logger = logging.getLogger(__name__)


class NoiseSuppressionError(Exception):
    """Exception raised by noise suppression operations."""
    pass


class NoiseSupressor:
    """Noise suppression for audio streams using noisereduce or RNNoise."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize noise suppressor.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.enabled = HAS_NOISEREDUCE or HAS_RNNOISE
        self.use_noisereduce = HAS_NOISEREDUCE
        self.use_rnnoise = HAS_RNNOISE and not HAS_NOISEREDUCE  # Prefer noisereduce
        
        # RNNoise specific settings
        self.target_sample_rate = 48000  # RNNoise works at 48kHz
        self.frame_size = 480  # 10ms at 48kHz
        
        # RNNoise state
        self._rnnoise_state: Optional[any] = None
        self._initialized = False
        
        # Resampling buffers for RNNoise
        self._input_buffer = np.array([], dtype=np.float32)
        self._output_buffer = np.array([], dtype=np.float32)
        
        # Noise profile for noisereduce (adaptive)
        self._noise_profile = None
        self._profile_frames = 0
        self._max_profile_frames = 50  # Collect first 50 frames for noise profile
        
        if self.enabled:
            if self.use_rnnoise:
                self._initialize_rnnoise()
            elif self.use_noisereduce:
                self._test_noisereduce()
            else:
                logger.info("Using noisereduce for noise suppression")
                self._initialized = True
    
    def _initialize_rnnoise(self) -> None:
        """Initialize RNNoise state (fallback)."""
        if not HAS_RNNOISE:
            raise NoiseSuppressionError("RNNoise not available")
        
        try:
            # Create RNNoise state
            self._rnnoise_state = rnnoise.RNNoise()
            self._initialized = True
            logger.info("RNNoise initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RNNoise: {e}")
            self.enabled = False
            raise NoiseSuppressionError(f"RNNoise initialization failed: {e}")
    
    def _test_noisereduce(self) -> None:
        """Test noisereduce functionality to ensure it works correctly."""
        try:
            # Test with a small audio sample
            test_audio = np.random.normal(0, 0.1, 2048).astype(np.float32)
            
            # Try basic noise reduction
            nr.reduce_noise(
                y=test_audio,
                sr=self.sample_rate,
                stationary=False,
                prop_decrease=0.6,
                n_fft=1024,
                win_length=512,
                hop_length=256,
                n_jobs=1
            )
            
            self._initialized = True
            logger.info("noisereduce test successful - noise suppression enabled")
            
        except Exception as e:
            logger.warning(f"noisereduce test failed - disabling noise suppression: {e}")
            self.enabled = False
            self.use_noisereduce = False
            self._initialized = False
    
    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply noise suppression to audio data.
        
        Args:
            audio_data: Input audio samples (float32, normalized to [-1, 1])
            
        Returns:
            Noise-suppressed audio data
        """
        if not self.enabled or not self._initialized:
            # Return original data if noise suppression is disabled
            return audio_data
        
        try:
            if self.use_noisereduce:
                return self._process_with_noisereduce(audio_data)
            elif self.use_rnnoise:
                return self._process_with_rnnoise(audio_data)
            else:
                return audio_data
                
        except Exception as e:
            logger.error(f"Error in noise suppression: {e}")
            # Return original data on error
            return audio_data
    
    def _process_with_noisereduce(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio using noisereduce.
        
        Args:
            audio_data: Input audio samples
            
        Returns:
            Processed audio data
        """
        if len(audio_data) == 0:
            return audio_data
        
        # Ensure minimum chunk size for noisereduce
        if len(audio_data) < 1024:
            return audio_data
        
        # Build noise profile from initial frames
        if self._noise_profile is None and self._profile_frames < self._max_profile_frames:
            if self._profile_frames == 0:
                self._noise_profile = audio_data.copy()
            else:
                # Accumulate noise samples
                self._noise_profile = np.concatenate([self._noise_profile, audio_data])
            
            self._profile_frames += 1
            
            # Don't apply suppression until we have a noise profile
            if self._profile_frames < self._max_profile_frames:
                return audio_data
        
        # Apply noise reduction
        try:
            # Use stationary noise reduction with collected noise profile
            if self._noise_profile is not None:
                # Ensure noise profile is long enough
                if len(self._noise_profile) < 1024:
                    return audio_data
                
                reduced_noise = nr.reduce_noise(
                    y=audio_data, 
                    sr=self.sample_rate,
                    y_noise=self._noise_profile,
                    stationary=True,
                    prop_decrease=0.6,  # Reduce noise by 60% (less aggressive)
                    n_fft=1024,  # Explicit FFT size
                    win_length=512,  # Window length
                    hop_length=256,  # Hop length (ensure overlap < win_length)
                    n_jobs=1  # Single threaded to avoid issues
                )
            else:
                # Fallback to non-stationary if no noise profile
                reduced_noise = nr.reduce_noise(
                    y=audio_data, 
                    sr=self.sample_rate,
                    stationary=False,
                    prop_decrease=0.6,
                    n_fft=1024,
                    win_length=512,
                    hop_length=256,
                    n_jobs=1
                )
            
            return reduced_noise.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in noisereduce processing: {e}")
            # Reset noise profile on error and disable processing for this session
            self._noise_profile = None
            self._profile_frames = self._max_profile_frames  # Skip profiling
            return audio_data
    
    def _process_with_rnnoise(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio using RNNoise (original implementation).
        
        Args:
            audio_data: Input audio samples
            
        Returns:
            Processed audio data
        """
        # Convert to the format RNNoise expects
        if self.sample_rate != self.target_sample_rate:
            # Resample to 48kHz if needed
            audio_48k = self._resample_to_48k(audio_data)
        else:
            audio_48k = audio_data
        
        # Process in 10ms frames (480 samples at 48kHz)
        processed_frames = []
        
        # Add to buffer
        self._input_buffer = np.concatenate([self._input_buffer, audio_48k])
        
        # Process complete frames
        while len(self._input_buffer) >= self.frame_size:
            # Extract frame
            frame = self._input_buffer[:self.frame_size]
            self._input_buffer = self._input_buffer[self.frame_size:]
            
            # Apply RNNoise
            processed_frame = self._process_frame(frame)
            processed_frames.append(processed_frame)
        
        if not processed_frames:
            return np.array([], dtype=np.float32)
        
        # Combine processed frames
        processed_audio = np.concatenate(processed_frames)
        
        # Resample back to original sample rate if needed
        if self.sample_rate != self.target_sample_rate:
            processed_audio = self._resample_from_48k(processed_audio)
        
        return processed_audio
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with RNNoise.
        
        Args:
            frame: Audio frame (480 samples at 48kHz)
            
        Returns:
            Processed frame
        """
        if not self._rnnoise_state:
            return frame
        
        try:
            # Convert to int16 for RNNoise
            frame_int16 = (frame * 32767).astype(np.int16)
            
            # Apply RNNoise
            processed_int16 = self._rnnoise_state.process(frame_int16)
            
            # Convert back to float32
            processed_float = processed_int16.astype(np.float32) / 32767.0
            
            return processed_float
            
        except Exception as e:
            logger.error(f"Error processing frame with RNNoise: {e}")
            return frame
    
    def _resample_to_48k(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Resample audio to 48kHz.
        
        Args:
            audio_data: Input audio at original sample rate
            
        Returns:
            Audio resampled to 48kHz
        """
        if len(audio_data) == 0:
            return audio_data
        
        # Simple linear interpolation resampling
        # For production, would use scipy.signal.resample or librosa
        ratio = self.target_sample_rate / self.sample_rate
        new_length = int(len(audio_data) * ratio)
        
        if new_length == 0:
            return np.array([], dtype=np.float32)
        
        # Create new time indices
        old_indices = np.arange(len(audio_data))
        new_indices = np.linspace(0, len(audio_data) - 1, new_length)
        
        # Interpolate
        resampled = np.interp(new_indices, old_indices, audio_data)
        
        return resampled.astype(np.float32)
    
    def _resample_from_48k(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Resample audio from 48kHz back to original sample rate.
        
        Args:
            audio_data: Input audio at 48kHz
            
        Returns:
            Audio resampled to original sample rate
        """
        if len(audio_data) == 0:
            return audio_data
        
        # Simple linear interpolation resampling
        ratio = self.sample_rate / self.target_sample_rate
        new_length = int(len(audio_data) * ratio)
        
        if new_length == 0:
            return np.array([], dtype=np.float32)
        
        # Create new time indices
        old_indices = np.arange(len(audio_data))
        new_indices = np.linspace(0, len(audio_data) - 1, new_length)
        
        # Interpolate
        resampled = np.interp(new_indices, old_indices, audio_data)
        
        return resampled.astype(np.float32)
    
    def reset(self) -> None:
        """Reset noise suppressor state."""
        if self.use_rnnoise and self._rnnoise_state:
            try:
                # Reset RNNoise state
                self._rnnoise_state = rnnoise.RNNoise()
                self._input_buffer = np.array([], dtype=np.float32)
                self._output_buffer = np.array([], dtype=np.float32)
                logger.debug("RNNoise suppressor reset")
            except Exception as e:
                logger.error(f"Error resetting RNNoise suppressor: {e}")
        
        if self.use_noisereduce:
            # Reset noise profile for noisereduce
            self._noise_profile = None
            self._profile_frames = 0
            logger.debug("Noisereduce suppressor reset")
    
    def is_enabled(self) -> bool:
        """Check if noise suppression is enabled and available."""
        return self.enabled and self._initialized
    
    def get_stats(self) -> dict:
        """Get noise suppressor statistics."""
        stats = {
            'enabled': self.enabled,
            'initialized': self._initialized,
            'has_noisereduce': HAS_NOISEREDUCE,
            'has_rnnoise': HAS_RNNOISE,
            'using_noisereduce': self.use_noisereduce,
            'using_rnnoise': self.use_rnnoise,
            'sample_rate': self.sample_rate,
        }
        
        if self.use_rnnoise:
            stats.update({
                'target_sample_rate': self.target_sample_rate,
                'frame_size': self.frame_size,
                'input_buffer_size': len(self._input_buffer),
                'output_buffer_size': len(self._output_buffer)
            })
        
        if self.use_noisereduce:
            stats.update({
                'noise_profile_frames': self._profile_frames,
                'has_noise_profile': self._noise_profile is not None
            })
        
        return stats
    
    def __del__(self):
        """Cleanup resources."""
        if self._rnnoise_state:
            try:
                del self._rnnoise_state
            except:
                pass


# Convenience function for simple usage
def suppress_noise(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Apply noise suppression to audio data (convenience function).
    
    Args:
        audio_data: Input audio samples
        sample_rate: Audio sample rate
        
    Returns:
        Noise-suppressed audio data
    """
    suppressor = NoiseSupressor(sample_rate)
    return suppressor.process(audio_data) 