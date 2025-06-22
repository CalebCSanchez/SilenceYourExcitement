"""
Real-time audio capture using sounddevice.
Handles microphone input with minimal latency for voice detection.
"""

import logging
import time
import threading
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


@dataclass
class AudioDevice:
    """Represents an audio input device."""
    index: int
    name: str
    channels: int
    sample_rate: float
    is_default: bool = False


class AudioCapture:
    """Real-time audio capture with configurable processing."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 320,  # ~20ms at 16kHz
        device_index: Optional[int] = None,
        on_audio_callback: Optional[Callable[[np.ndarray], None]] = None
    ):
        """
        Initialize audio capture.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of samples per chunk
            device_index: Input device index (None for default)
            on_audio_callback: Callback for processed audio data
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.on_audio_callback = on_audio_callback
        
        # State
        self._stream: Optional[sd.InputStream] = None
        self._is_running = False
        self._capture_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.frames_processed = 0
        self.start_time = 0.0
        self.last_callback_time = 0.0
        self.callback_errors = 0
        
        # Audio buffer for analysis
        self._audio_buffer = np.zeros(chunk_size, dtype=np.float32)
        self._buffer_lock = threading.Lock()
    
    @staticmethod
    def get_available_devices() -> List[AudioDevice]:
        """
        Get list of available audio input devices.
        
        Returns:
            List of available input devices
        """
        devices = []
        
        try:
            device_list = sd.query_devices()
            default_input = sd.default.device[0] if sd.default.device else None
            
            for i, device_info in enumerate(device_list):
                # Only include input devices
                if device_info['max_input_channels'] > 0:
                    device = AudioDevice(
                        index=i,
                        name=device_info['name'],
                        channels=device_info['max_input_channels'],
                        sample_rate=device_info['default_samplerate'],
                        is_default=(i == default_input)
                    )
                    devices.append(device)
            
            logger.info(f"Found {len(devices)} input devices")
            
        except Exception as e:
            logger.error(f"Error querying audio devices: {e}")
        
        return devices
    
    @staticmethod
    def get_default_device() -> Optional[AudioDevice]:
        """Get the default input device."""
        devices = AudioCapture.get_available_devices()
        for device in devices:
            if device.is_default:
                return device
        return devices[0] if devices else None
    
    def start_capture(self) -> bool:
        """
        Start audio capture.
        
        Returns:
            True if capture started successfully
        """
        if self._is_running:
            logger.warning("Audio capture already running")
            return True
        
        try:
            # Validate device
            if self.device_index is not None:
                device_info = sd.query_devices(self.device_index)
                if device_info['max_input_channels'] == 0:
                    logger.error(f"Device {self.device_index} is not an input device")
                    return False
            
            # Create audio stream
            self._stream = sd.InputStream(
                device=self.device_index,
                channels=1,  # Mono
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.float32,
                callback=self._audio_callback,
                finished_callback=self._stream_finished_callback
            )
            
            # Start stream
            self._stream.start()
            self._is_running = True
            self.start_time = time.time()
            self.frames_processed = 0
            self.callback_errors = 0
            
            device_name = "default"
            if self.device_index is not None:
                device_name = sd.query_devices(self.device_index)['name']
            
            logger.info(f"Audio capture started - Device: {device_name}, "
                       f"Sample rate: {self.sample_rate}, Chunk size: {self.chunk_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            self._cleanup_stream()
            return False
    
    def stop_capture(self) -> None:
        """Stop audio capture."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
        
        self._cleanup_stream()
        
        # Log statistics
        if self.start_time > 0:
            duration = time.time() - self.start_time
            fps = self.frames_processed / duration if duration > 0 else 0
            logger.info(f"Audio capture stopped - Duration: {duration:.1f}s, "
                       f"Frames: {self.frames_processed}, FPS: {fps:.1f}, "
                       f"Errors: {self.callback_errors}")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        """
        Audio stream callback.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Timing information
            status: Stream status flags
        """
        try:
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            if not self._is_running:
                return
            
            # Convert to mono if needed
            audio_data = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            
            # Update buffer
            with self._buffer_lock:
                self._audio_buffer = audio_data.copy()
            
            # Update statistics
            self.frames_processed += 1
            self.last_callback_time = time.time()
            
            # Call user callback
            if self.on_audio_callback:
                try:
                    self.on_audio_callback(audio_data)
                except Exception as e:
                    logger.error(f"Error in audio callback: {e}")
                    self.callback_errors += 1
                    
        except Exception as e:
            logger.error(f"Error in audio stream callback: {e}")
            self.callback_errors += 1
    
    def _stream_finished_callback(self) -> None:
        """Called when stream finishes."""
        logger.debug("Audio stream finished")
        self._is_running = False
    
    def _cleanup_stream(self) -> None:
        """Clean up stream resources."""
        self._stream = None
        self._is_running = False
    
    def get_latest_audio(self) -> np.ndarray:
        """
        Get the most recent audio buffer.
        
        Returns:
            Latest audio data
        """
        with self._buffer_lock:
            return self._audio_buffer.copy()
    
    def is_running(self) -> bool:
        """Check if capture is currently running."""
        return self._is_running
    
    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics."""
        duration = time.time() - self.start_time if self.start_time > 0 else 0
        fps = self.frames_processed / duration if duration > 0 else 0
        
        return {
            'is_running': self._is_running,
            'frames_processed': self.frames_processed,
            'duration_seconds': duration,
            'fps': fps,
            'callback_errors': self.callback_errors,
            'last_callback_time': self.last_callback_time,
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'device_index': self.device_index
        }
    
    def set_device(self, device_index: Optional[int]) -> bool:
        """
        Change audio input device.
        
        Args:
            device_index: New device index
            
        Returns:
            True if device changed successfully
        """
        was_running = self._is_running
        
        if was_running:
            self.stop_capture()
        
        self.device_index = device_index
        
        if was_running:
            return self.start_capture()
        
        return True
    
    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture() 