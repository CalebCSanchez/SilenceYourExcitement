"""
Configuration management for Rage-Reducer.
Handles loading and saving user settings to JSON.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio-related configuration settings."""
    device_index: Optional[int] = None
    device_name: str = ""
    sample_rate: int = 16000
    chunk_size: int = 320  # ~20ms at 16kHz
    baseline_volume: float = -30.0
    warning_threshold: float = -27.0
    angry_threshold: float = -24.0


@dataclass
class OverlayConfig:
    """Overlay visual configuration."""
    animation_duration: int = 500  # ms
    update_interval: int = 50  # ms
    warning_opacity: float = 0.4
    angry_opacity: float = 0.85
    color_red: int = 255
    color_green: int = 0
    color_blue: int = 0


@dataclass
class GameConfig:
    """Game-specific configuration."""
    selected_game: str = "[Always active]"
    check_interval: int = 500  # ms
    enabled: bool = False


@dataclass
class AppConfig:
    """General application configuration."""
    start_with_windows: bool = False
    minimize_to_tray: bool = True
    show_notifications: bool = True
    log_level: str = "INFO"


class Config:
    """Main configuration manager."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to config file. If None, uses default location.
        """
        if config_file is None:
            config_dir = Path.home() / ".rage_reducer"
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / "config.json"
        
        self.config_file = config_file
        
        # Initialize with defaults
        self.audio = AudioConfig()
        self.overlay = OverlayConfig()
        self.game = GameConfig()
        self.app = AppConfig()
        
        # Load existing config
        self.load()
    
    def load(self) -> None:
        """Load configuration from file."""
        if not self.config_file.exists():
            logger.info(f"Config file not found: {self.config_file}. Using defaults.")
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update configuration sections
            if 'audio' in data:
                self._update_dataclass(self.audio, data['audio'])
            
            if 'overlay' in data:
                self._update_dataclass(self.overlay, data['overlay'])
            
            if 'game' in data:
                self._update_dataclass(self.game, data['game'])
            
            if 'app' in data:
                self._update_dataclass(self.app, data['app'])
            
            logger.info(f"Configuration loaded from {self.config_file}")
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load config: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}")
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            data = {
                'audio': asdict(self.audio),
                'overlay': asdict(self.overlay),
                'game': asdict(self.game),
                'app': asdict(self.app)
            }
            
            # Ensure config directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _update_dataclass(self, obj: Any, data: Dict[str, Any]) -> None:
        """Update dataclass fields from dictionary."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
            else:
                logger.warning(f"Unknown config key: {key}")
    
    def update_audio_thresholds(self, baseline: float) -> None:
        """Update audio thresholds based on new baseline."""
        # Convert to native Python float to avoid numpy serialization issues
        self.audio.baseline_volume = float(baseline)
        self.audio.warning_threshold = float(baseline + 3.0)
        self.audio.angry_threshold = float(baseline + 6.0)
        self.save()
        logger.info(f"Audio thresholds updated - baseline: {baseline:.1f}dB")
    
    def set_audio_device(self, device_index: int, device_name: str) -> None:
        """Set selected audio device."""
        self.audio.device_index = device_index
        self.audio.device_name = device_name
        self.save()
        logger.info(f"Audio device set to: {device_name} (index: {device_index})")
    
    def set_selected_game(self, game_name: str) -> None:
        """Set selected game for monitoring."""
        self.game.selected_game = game_name
        self.game.enabled = game_name != "[Always active]"
        self.save()
        logger.info(f"Selected game: {game_name}")
    
    def toggle_startup(self, enabled: bool) -> None:
        """Toggle Windows startup setting."""
        self.app.start_with_windows = enabled
        self.save()
        logger.info(f"Start with Windows: {enabled}")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return {
            'audio': asdict(self.audio),
            'overlay': asdict(self.overlay), 
            'game': asdict(self.game),
            'app': asdict(self.app)
        }
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to default values."""
        self.audio = AudioConfig()
        self.overlay = OverlayConfig()
        self.game = GameConfig()
        self.app = AppConfig()
        self.save()
        logger.info("Configuration reset to defaults") 