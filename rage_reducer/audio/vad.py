"""
Voice Activity Detection using Silero-VAD ONNX model.
Detects speech segments in audio to ignore non-speech noise.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple
import requests

import numpy as np

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    # Note: We don't log the warning here since it may be misleading during import
    # The actual availability will be checked during VAD initialization

logger = logging.getLogger(__name__)


class VADError(Exception):
    """Exception raised by VAD operations."""
    pass


class SileroVAD:
    """Silero Voice Activity Detection using ONNX."""
    
    # Model configuration - Updated to correct URLs
    MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    MODEL_FALLBACK_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
    MODEL_SAMPLE_RATE = 16000
    
    def __init__(self, model_path: Optional[Path] = None, threshold: float = 0.5):
        """
        Initialize Silero VAD.
        
        Args:
            model_path: Path to ONNX model file (downloads if None)
            threshold: Voice detection threshold (0.0-1.0)
        """
        self.threshold = threshold
        self.sample_rate = self.MODEL_SAMPLE_RATE
        self.enabled = HAS_ONNX
        
        # Model state
        self._session: Optional[ort.InferenceSession] = None
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._initialized = False
        
        # Statistics
        self.frames_processed = 0
        self.voice_frames = 0
        self.silence_frames = 0
        
        if self.enabled:
            # Set model path
            if model_path is None:
                model_dir = Path.home() / ".rage_reducer" / "models"
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / "silero_vad.onnx"
            
            self.model_path = model_path
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the ONNX model."""
        if not HAS_ONNX:
            raise VADError("ONNX Runtime not available")
        
        try:
            # Download model if it doesn't exist
            if not self.model_path.exists():
                logger.info(f"Downloading Silero VAD model to {self.model_path}")
                self._download_model()
            
            # Create ONNX session
            providers = ['CPUExecutionProvider']
            
            # Try GPU providers
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
                logger.info("Using CUDA for VAD inference")
            elif 'DmlExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'DmlExecutionProvider')
                logger.info("Using DirectML for VAD inference")
            
            self._session = ort.InferenceSession(str(self.model_path), providers=providers)
            self._initialized = True
            
            logger.info(f"Silero VAD initialized - Threshold: {self.threshold}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Silero VAD: {e}")
            self.enabled = False
            raise VADError(f"VAD initialization failed: {e}")
    
    def _download_model(self) -> None:
        """Download the Silero VAD model."""
        urls_to_try = [self.MODEL_URL, self.MODEL_FALLBACK_URL]
        
        for i, url in enumerate(urls_to_try):
            try:
                logger.info(f"Attempting to download VAD model from: {url}")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(self.model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Model downloaded successfully from {url}: {self.model_path}")
                return
                
            except Exception as e:
                logger.error(f"Failed to download VAD model from {url}: {e}")
                if i == len(urls_to_try) - 1:  # Last URL failed
                    raise VADError(f"All model download attempts failed. Last error: {e}")
                else:
                    logger.info(f"Trying next URL...")
                    continue
    
    def detect_voice(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect voice activity in audio data.
        
        Args:
            audio_data: Audio samples (float32, 16kHz)
            
        Returns:
            Tuple of (has_voice, confidence_score)
        """
        if not self.enabled or not self._initialized:
            # Always return True if VAD is disabled
            return True, 1.0
        
        try:
            # Ensure correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Ensure correct length for model (512 samples for Silero VAD)
            expected_length = 512
            if len(audio_data) != expected_length:
                if len(audio_data) < expected_length:
                    # Pad with zeros
                    padded = np.zeros(expected_length, dtype=np.float32)
                    padded[:len(audio_data)] = audio_data
                    audio_data = padded
                else:
                    # Truncate
                    audio_data = audio_data[:expected_length]
            
            # Prepare input
            input_data = audio_data.reshape(1, -1)
            
            # Run inference
            ort_inputs = {
                'input': input_data,
                'h': self._h,
                'c': self._c
            }
            
            ort_outputs = self._session.run(None, ort_inputs)
            output, h_out, c_out = ort_outputs
            
            # Update hidden states
            self._h = h_out
            self._c = c_out
            
            # Get voice probability
            voice_prob = float(output[0][0])
            has_voice = bool(voice_prob >= self.threshold)
            
            # Update statistics
            self.frames_processed += 1
            if has_voice:
                self.voice_frames += 1
            else:
                self.silence_frames += 1
            
            return has_voice, voice_prob
            
        except Exception as e:
            logger.error(f"Error in voice detection: {e}")
            # Return True on error to avoid blocking audio processing
            return True, 0.5
    
    def process_chunk(self, audio_chunk: np.ndarray, chunk_size: int = 512) -> list[Tuple[bool, float]]:
        """
        Process audio chunk in smaller frames for VAD.
        
        Args:
            audio_chunk: Audio data
            chunk_size: Size of each VAD frame
            
        Returns:
            List of (has_voice, confidence) for each frame
        """
        if not self.enabled:
            return [(True, 1.0)]
        
        results = []
        
        # Process in overlapping windows
        for i in range(0, len(audio_chunk), chunk_size // 2):
            frame = audio_chunk[i:i + chunk_size]
            if len(frame) < chunk_size // 2:  # Skip very short frames
                break
            
            has_voice, confidence = self.detect_voice(frame)
            results.append((has_voice, confidence))
        
        return results
    
    def reset(self) -> None:
        """Reset VAD internal state."""
        if self._initialized:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
            logger.debug("VAD state reset")
    
    def is_enabled(self) -> bool:
        """Check if VAD is enabled and available."""
        return self.enabled and self._initialized
    
    def get_stats(self) -> dict:
        """Get VAD statistics."""
        total_frames = self.frames_processed
        voice_ratio = self.voice_frames / total_frames if total_frames > 0 else 0
        
        return {
            'enabled': self.enabled,
            'initialized': self._initialized,
            'has_onnx': HAS_ONNX,
            'threshold': self.threshold,
            'frames_processed': self.frames_processed,
            'voice_frames': self.voice_frames,
            'silence_frames': self.silence_frames,
            'voice_ratio': voice_ratio,
            'model_path': str(self.model_path) if hasattr(self, 'model_path') else None
        }
    
    def set_threshold(self, threshold: float) -> None:
        """Set voice detection threshold."""
        self.threshold = max(0.0, min(1.0, threshold))
        logger.info(f"VAD threshold set to {self.threshold}")


class SimpleVAD:
    """Simple energy-based VAD as fallback when Silero is not available."""
    
    def __init__(self, threshold: float = -40.0, sample_rate: int = 16000):
        """
        Initialize simple VAD.
        
        Args:
            threshold: Energy threshold in dBFS
            sample_rate: Audio sample rate
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.enabled = True
        
        # Statistics
        self.frames_processed = 0
        self.voice_frames = 0
        self.silence_frames = 0
    
    def detect_voice(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect voice using simple energy threshold.
        
        Args:
            audio_data: Audio samples
            
        Returns:
            Tuple of (has_voice, energy_db)
        """
        if len(audio_data) == 0:
            return False, -np.inf
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms == 0:
            energy_db = -np.inf
        else:
            energy_db = 20 * np.log10(rms)
        
        has_voice = bool(energy_db > self.threshold)
        
        # Update statistics
        self.frames_processed += 1
        if has_voice:
            self.voice_frames += 1
        else:
            self.silence_frames += 1
        
        return has_voice, energy_db
    
    def reset(self) -> None:
        """Reset simple VAD state."""
        pass  # No state to reset
    
    def is_enabled(self) -> bool:
        """Check if simple VAD is enabled."""
        return self.enabled
    
    def get_stats(self) -> dict:
        """Get simple VAD statistics."""
        total_frames = self.frames_processed
        voice_ratio = self.voice_frames / total_frames if total_frames > 0 else 0
        
        return {
            'type': 'simple',
            'enabled': self.enabled,
            'threshold': self.threshold,
            'frames_processed': self.frames_processed,
            'voice_frames': self.voice_frames,
            'silence_frames': self.silence_frames,
            'voice_ratio': voice_ratio
        }


# Factory function
def create_vad(use_silero: bool = True, threshold: float = 0.5) -> SileroVAD | SimpleVAD:
    """
    Create VAD instance.
    
    Args:
        use_silero: Try to use Silero VAD first
        threshold: Detection threshold
        
    Returns:
        VAD instance (Silero or Simple fallback)
    """
    if use_silero and HAS_ONNX:
        try:
            vad = SileroVAD(threshold=threshold)
            logger.info("Silero VAD initialized successfully")
            return vad
        except VADError as e:
            logger.warning(f"Silero VAD initialization failed: {e}")
            logger.warning("Falling back to simple VAD")
    elif use_silero and not HAS_ONNX:
        logger.warning("ONNX Runtime not available - using simple VAD instead of Silero VAD")
    
    # Fallback to simple VAD
    # Use a more permissive threshold for simple VAD to ensure voice detection
    simple_threshold = -50.0 if use_silero else threshold
    vad = SimpleVAD(threshold=simple_threshold)
    logger.info("Simple VAD initialized successfully")
    return vad 