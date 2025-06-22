"""
Main settings window for Rage-Reducer.
Provides user interface for all configuration options.
"""

import logging
import time
from typing import Optional, List
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QCheckBox, QSlider, QProgressBar,
    QGroupBox, QStatusBar, QMessageBox, QSystemTrayIcon, QMenu,
    QApplication, QFrame, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QFont, QPixmap, QAction

from ..core.config import Config
from ..core.state import AudioState, VolumeState, RageLevel, calculate_rms_db, calculate_baseline_from_samples
from ..core.gamecheck import GameChecker
from ..audio.capture import AudioCapture, AudioDevice
from ..audio.suppress import NoiseSupressor
from ..audio.vad import create_vad
from .overlay import MultiScreenOverlayManager
from .tray import TrayManager

logger = logging.getLogger(__name__)


class AudioProcessingThread(QThread):
    """Background thread for audio processing."""
    
    # Signals
    volume_updated = pyqtSignal(float, bool)  # volume_db, has_voice
    state_changed = pyqtSignal(str)  # state name
    rage_changed = pyqtSignal(str, float, float)  # rage_level, rage_fill, target_opacity
    calibration_sample = pyqtSignal(float)  # calibration volume sample
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.running = False
        self.calibrating = False
        
        # Audio components
        self.audio_capture: Optional[AudioCapture] = None
        self.noise_suppressor: Optional[NoiseSupressor] = None
        self.vad = create_vad()
        self.audio_state = AudioState()
        
        # Game checking
        self.game_checker = GameChecker()
        
        # Statistics
        self.frames_processed = 0
        self.start_time = 0.0
        
        # Calibration
        self.calibration_samples: List[float] = []
        self.calibration_start_time = 0.0
        self.calibration_duration = 5.0  # seconds
        
        # Setup audio state callbacks
        self.audio_state.on_state_change = self._on_state_change
        self.audio_state.on_rage_change = self._on_rage_change
    
    def start_processing(self) -> bool:
        """Start audio processing."""
        if self.running:
            return True
        
        try:
            # Initialize audio capture
            self.audio_capture = AudioCapture(
                sample_rate=self.config.audio.sample_rate,
                chunk_size=self.config.audio.chunk_size,
                device_index=self.config.audio.device_index,
                on_audio_callback=self._process_audio_frame
            )
            
            # Initialize noise suppressor (optional)
            try:
                self.noise_suppressor = NoiseSupressor(self.config.audio.sample_rate)
                if not self.noise_suppressor.is_enabled():
                    logger.info("Noise suppression disabled - continuing without it")
                    self.noise_suppressor = None
            except Exception as e:
                logger.warning(f"Failed to initialize noise suppressor: {e}")
                self.noise_suppressor = None
            
            # Update audio state thresholds
            self.audio_state.thresholds.from_baseline(self.config.audio.baseline_volume)
            
            # Start audio capture
            if not self.audio_capture.start_capture():
                return False
            
            self.running = True
            self.start_time = time.time()
            self.frames_processed = 0
            
            # Start thread
            self.start()
            
            logger.info("Audio processing started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio processing: {e}")
            return False
    
    def stop_processing(self) -> None:
        """Stop audio processing."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop audio capture
        if self.audio_capture:
            self.audio_capture.stop_capture()
        
        # Wait for thread to finish
        self.wait(5000)  # 5 second timeout
        
        logger.info("Audio processing stopped")
    
    def start_calibration(self) -> None:
        """Start voice calibration."""
        self.calibrating = True
        self.calibration_samples.clear()
        self.calibration_start_time = time.time()
        logger.info("Calibration started")
    
    def stop_calibration(self) -> float:
        """Stop calibration and return baseline volume."""
        self.calibrating = False
        
        if not self.calibration_samples:
            logger.warning("No calibration samples collected")
            return self.config.audio.baseline_volume
        
        # Calculate baseline
        baseline = calculate_baseline_from_samples(self.calibration_samples)
        
        # Update configuration
        self.config.update_audio_thresholds(baseline)
        
        # Update audio state
        self.audio_state.thresholds.from_baseline(baseline)
        
        logger.info(f"Calibration completed - Baseline: {baseline:.1f}dB")
        return baseline
    
    def run(self) -> None:
        """Main thread loop."""
        while self.running:
            # Check if game-specific monitoring is enabled
            if self.config.game.enabled:
                game_active = self.game_checker.is_game_foreground(self.config.game.selected_game)
                if not game_active:
                    # Game not in foreground, let rage meter decay naturally instead of resetting
                    # The rage meter will automatically decay when no new audio triggers it
                    pass
            
            # Sleep briefly to avoid busy waiting
            self.msleep(100)
    
    def _process_audio_frame(self, audio_data) -> None:
        """Process audio frame from capture callback."""
        try:
            # Apply noise suppression
            if self.noise_suppressor and self.noise_suppressor.is_enabled():
                audio_data = self.noise_suppressor.process(audio_data)
            
            # Voice activity detection
            has_voice, vad_confidence = self.vad.detect_voice(audio_data)
            
            # Calculate volume
            volume_db = calculate_rms_db(audio_data)
            
            # Debug: Log voice detection issues
            if volume_db > -35.0 and not has_voice:
                logger.warning(f"Voice volume high ({volume_db:.1f}dB) but VAD not detecting voice (confidence: {vad_confidence:.2f})")
            elif volume_db > -25.0 and has_voice:
                logger.debug(f"Voice detected: {volume_db:.1f}dB, VAD confidence: {vad_confidence:.2f}")
            
            # Handle calibration
            if self.calibrating:
                if has_voice and volume_db > -60.0:  # Only collect voice samples
                    self.calibration_samples.append(volume_db)
                    self.calibration_sample.emit(volume_db)
                
                # Check calibration timeout
                if time.time() - self.calibration_start_time >= self.calibration_duration:
                    self.stop_calibration()
                
                return
            
            # Update audio state
            if not self.calibrating:
                current_state = self.audio_state.update_volume(volume_db, has_voice)
            
            # Emit signals (convert numpy types to Python native types)
            self.volume_updated.emit(float(volume_db), bool(has_voice))
            
            self.frames_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
    
    def _on_rage_change(self, previous_rage: RageLevel, current_rage: RageLevel, rage_fill: float) -> None:
        """Handle rage level changes."""
        target_opacity = self.audio_state.get_rage_opacity()
        self.rage_changed.emit(current_rage.value, rage_fill, target_opacity)
    
    def _on_state_change(self, previous_state: VolumeState, current_state: VolumeState) -> None:
        """Handle audio state changes."""
        self.state_changed.emit(current_state.value)
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        duration = time.time() - self.start_time if self.start_time > 0 else 0
        fps = self.frames_processed / duration if duration > 0 else 0
        
        stats = {
            'running': self.running,
            'calibrating': self.calibrating,
            'frames_processed': self.frames_processed,
            'duration_seconds': duration,
            'fps': fps,
            'audio_state': self.audio_state.current_state.value,
            'calibration_samples': len(self.calibration_samples)
        }
        
        if self.audio_capture:
            stats['capture'] = self.audio_capture.get_stats()
        
        if self.noise_suppressor:
            stats['noise_suppressor'] = self.noise_suppressor.get_stats()
        
        stats['vad'] = self.vad.get_stats()
        
        return stats


class MainSettingsWindow(QMainWindow):
    """Main settings window for Rage-Reducer."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Components
        self.audio_thread: Optional[AudioProcessingThread] = None
        self.overlay_manager = MultiScreenOverlayManager(primary_only=True)  # Only show on primary monitor
        self.tray_manager: Optional[TrayManager] = None
        self.game_checker = GameChecker()
        
        # UI elements
        self.device_combo: Optional[QComboBox] = None
        self.game_combo: Optional[QComboBox] = None
        self.calibrate_button: Optional[QPushButton] = None
        self.start_button: Optional[QPushButton] = None
        self.stop_button: Optional[QPushButton] = None
        self.startup_checkbox: Optional[QCheckBox] = None
        self.volume_bar: Optional[QProgressBar] = None
        self.state_label: Optional[QLabel] = None
        self.status_bar: Optional[QStatusBar] = None
        
        # Timers
        self.ui_update_timer = QTimer()
        self.ui_update_timer.timeout.connect(self._update_ui)
        self.ui_update_timer.start(100)  # Update UI every 100ms
        
        # Calibration
        self.calibration_progress: Optional[QProgressBar] = None
        
        self._setup_ui()
        self._setup_tray()
        self._load_settings()  # Now that the None check is fixed, this will work
    
    def _setup_ui(self) -> None:
        """Setup user interface."""
        logger.debug("Setting up UI")
        self.setWindowTitle("Rage-Reducer - Gaming Voice Control")
        self.setMinimumSize(600, 500)
        self.resize(700, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("🎮 Rage-Reducer")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Keep your gaming voice under control")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: gray; margin-bottom: 20px;")
        main_layout.addWidget(subtitle_label)
        
        # Audio settings group
        audio_group = self._create_audio_settings_group()
        main_layout.addWidget(audio_group)
        
        # Game settings group
        game_group = self._create_game_settings_group()
        main_layout.addWidget(game_group)
        
        # Status group
        status_group = self._create_status_group()
        main_layout.addWidget(status_group)
        
        # Control buttons
        control_layout = self._create_control_buttons()
        main_layout.addLayout(control_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to reduce rage!")
    
    def _create_audio_settings_group(self) -> QGroupBox:
        """Create audio settings group."""
        logger.debug("Creating audio settings group")
        group = QGroupBox("🎤 Audio Settings")
        layout = QGridLayout(group)
        
        # Microphone selection
        layout.addWidget(QLabel("Microphone:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        layout.addWidget(self.device_combo, 0, 1, 1, 2)
        logger.debug("Device combo created")
        
        # Calibration
        layout.addWidget(QLabel("Voice Calibration:"), 1, 0)
        self.calibrate_button = QPushButton("🎯 Calibrate (5s)")
        self.calibrate_button.clicked.connect(self._calibrate_voice)
        layout.addWidget(self.calibrate_button, 1, 1)
        
        self.calibration_progress = QProgressBar()
        self.calibration_progress.setVisible(False)
        layout.addWidget(self.calibration_progress, 1, 2)
        
        # Current volume display
        layout.addWidget(QLabel("Volume Level:"), 2, 0)
        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(-60, 0)  # dBFS range
        self.volume_bar.setValue(-30)
        layout.addWidget(self.volume_bar, 2, 1, 1, 2)
        
        # Current state
        layout.addWidget(QLabel("Current State:"), 3, 0)
        self.state_label = QLabel("😌 Calm")
        self.state_label.setStyleSheet("font-weight: bold; color: green;")
        layout.addWidget(self.state_label, 3, 1, 1, 2)
        
        return group
    
    def _create_game_settings_group(self) -> QGroupBox:
        """Create game settings group."""
        group = QGroupBox("🎮 Game Settings")
        layout = QGridLayout(group)
        
        # Game selection
        layout.addWidget(QLabel("Target Game:"), 0, 0)
        self.game_combo = QComboBox()
        self.game_combo.currentTextChanged.connect(self._on_game_changed)
        layout.addWidget(self.game_combo, 0, 1)
        logger.debug("Game combo created")
        
        refresh_button = QPushButton("🔄 Refresh")
        refresh_button.clicked.connect(self._refresh_games)
        layout.addWidget(refresh_button, 0, 2)
        
        # Windows startup
        self.startup_checkbox = QCheckBox("Start with Windows")
        self.startup_checkbox.toggled.connect(self._on_startup_toggled)
        layout.addWidget(self.startup_checkbox, 1, 0, 1, 3)
        
        return group
    
    def _create_status_group(self) -> QGroupBox:
        """Create status display group."""
        group = QGroupBox("📊 Status")
        layout = QVBoxLayout(group)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)
        
        return group
    
    def _create_control_buttons(self) -> QHBoxLayout:
        """Create control buttons layout."""
        layout = QHBoxLayout()
        
        self.start_button = QPushButton("▶️ Start Monitoring")
        self.start_button.clicked.connect(self._start_monitoring)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹️ Stop Monitoring")
        self.stop_button.clicked.connect(self._stop_monitoring)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
        layout.addWidget(self.stop_button)
        
        return layout
    
    def _setup_tray(self) -> None:
        """Setup system tray."""
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_manager = TrayManager(self)
            self.tray_manager.show()
    
    def _load_settings(self) -> None:
        """Load settings from configuration."""
        logger.debug(f"Loading settings - device_combo: {self.device_combo is not None}, game_combo: {self.game_combo is not None}")
        
        # Load audio devices
        self._refresh_audio_devices()
        
        # Load games
        self._refresh_games()
        
        # Load other settings
        if self.startup_checkbox:
            self.startup_checkbox.setChecked(self.config.app.start_with_windows)
    
    def _refresh_audio_devices(self) -> None:
        """Refresh audio device list."""
        if self.device_combo is None:
            logger.warning("Device combo is None")
            return
        
        self.device_combo.clear()
        
        try:
            devices = AudioCapture.get_available_devices()
            logger.info(f"Found {len(devices)} audio devices for dropdown")
            
            # Sort devices to put default first
            default_devices = [d for d in devices if d.is_default]
            other_devices = [d for d in devices if not d.is_default]
            sorted_devices = default_devices + other_devices
            
            for device in sorted_devices:
                display_name = f"{device.name}"
                if device.is_default:
                    display_name += " (Default)"
                
                self.device_combo.addItem(display_name, device.index)
                logger.debug(f"Added device: {display_name} (index: {device.index})")
            
            logger.info(f"Device combo now has {self.device_combo.count()} items")
            
            # Select configured device
            if self.config.audio.device_index is not None:
                for i in range(self.device_combo.count()):
                    if self.device_combo.itemData(i) == self.config.audio.device_index:
                        self.device_combo.setCurrentIndex(i)
                        logger.info(f"Selected configured device at index {i}")
                        break
        except Exception as e:
            logger.error(f"Error refreshing audio devices: {e}")
    
    def _refresh_games(self) -> None:
        """Refresh game list."""
        if self.game_combo is None:
            logger.warning("Game combo is None")
            return
        
        current_text = self.game_combo.currentText()
        self.game_combo.clear()
        
        try:
            games = self.game_checker.get_game_names()
            logger.info(f"Found {len(games)} games for dropdown")
            
            self.game_combo.addItems(games)
            logger.info(f"Game combo now has {self.game_combo.count()} items")
            
            # Restore previous selection
            index = self.game_combo.findText(self.config.game.selected_game)
            if index >= 0:
                self.game_combo.setCurrentIndex(index)
                logger.info(f"Selected configured game: {self.config.game.selected_game}")
            elif current_text:
                index = self.game_combo.findText(current_text)
                if index >= 0:
                    self.game_combo.setCurrentIndex(index)
                    logger.info(f"Restored previous selection: {current_text}")
        except Exception as e:
            logger.error(f"Error refreshing games: {e}")
    
    def _calibrate_voice(self) -> None:
        """Start voice calibration with automatic monitoring."""
        # Store whether monitoring was already running
        was_monitoring = self.audio_thread and self.audio_thread.running
        
        # Start monitoring if not already running
        if not was_monitoring:
            self._start_monitoring()
            if not (self.audio_thread and self.audio_thread.running):
                QMessageBox.warning(self, "Warning", "Failed to start monitoring for calibration!")
                return
        
        # Show progress bar
        if self.calibration_progress:
            self.calibration_progress.setVisible(True)
            self.calibration_progress.setRange(0, 50)  # 5 seconds * 10 updates/sec
            self.calibration_progress.setValue(0)
        
        # Disable calibrate button and update other UI
        if self.calibrate_button:
            self.calibrate_button.setEnabled(False)
            self.calibrate_button.setText("Calibrating...")
        
        # Disable start/stop buttons during calibration
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        # Store calibration state
        self._was_monitoring_before_calibration = was_monitoring
        
        # Start calibration
        self.audio_thread.start_calibration()
        
        # Update status
        self.status_bar.showMessage("Calibrating... Speak normally for 5 seconds")
    
    def _start_monitoring(self) -> None:
        """Start audio monitoring."""
        try:
            # Create audio thread
            self.audio_thread = AudioProcessingThread(self.config)
            
            # Connect signals
            self.audio_thread.volume_updated.connect(self._on_volume_updated)
            self.audio_thread.state_changed.connect(self._on_state_changed)
            self.audio_thread.rage_changed.connect(self._on_rage_changed)
            self.audio_thread.calibration_sample.connect(self._on_calibration_sample)
            
            # Start processing
            if self.audio_thread.start_processing():
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.status_bar.showMessage("Monitoring started - Voice levels being tracked")
                
                # Update tray
                if self.tray_manager:
                    self.tray_manager.set_state("monitoring")
            else:
                QMessageBox.critical(self, "Error", "Failed to start audio monitoring!")
                
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start monitoring: {e}")
    
    def _stop_monitoring(self) -> None:
        """Stop audio monitoring."""
        if self.audio_thread:
            self.audio_thread.stop_processing()
            self.audio_thread = None
        
        # Hide overlay
        self.overlay_manager.force_hide_all()
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_bar.showMessage("Monitoring stopped")
        
        # Reset state display
        if self.state_label:
            self.state_label.setText("😌 Calm")
            self.state_label.setStyleSheet("font-weight: bold; color: green;")
        
        # Update tray
        if self.tray_manager:
            self.tray_manager.set_state("idle")
    
    def _on_device_changed(self, index: int) -> None:
        """Handle audio device selection change."""
        device_index = self.device_combo.itemData(index)
        device_name = self.device_combo.itemText(index)
        
        self.config.set_audio_device(device_index, device_name)
    
    def _on_game_changed(self, game_name: str) -> None:
        """Handle game selection change."""
        self.config.set_selected_game(game_name)
    
    def _on_startup_toggled(self, checked: bool) -> None:
        """Handle startup checkbox toggle."""
        self.config.toggle_startup(checked)
        # TODO: Implement Windows startup registry modification
    
    def _on_volume_updated(self, volume_db: float, has_voice: bool) -> None:
        """Handle volume update from audio thread."""
        if self.volume_bar:
            # Clamp volume to display range
            display_volume = max(-60, min(0, int(volume_db)))
            self.volume_bar.setValue(display_volume)
    
    def _on_rage_changed(self, rage_level_name: str, rage_fill: float, target_opacity: float) -> None:
        """Handle rage level change from audio thread."""
        logger.info(f"UI: Rage changed to {rage_level_name}, fill: {rage_fill:.2f}, opacity: {target_opacity:.2f}")
        
        # Update state label based on rage level
        if self.state_label:
            if rage_level_name == "calm":
                self.state_label.setText("😌 Calm")
                self.state_label.setStyleSheet("font-weight: bold; color: green;")
            elif rage_level_name == "medium":
                self.state_label.setText(f"😐 Warning ({rage_fill:.1f})")
                self.state_label.setStyleSheet("font-weight: bold; color: orange;")
            elif rage_level_name == "angry":
                self.state_label.setText(f"😠 Angry ({rage_fill:.1f})")
                self.state_label.setStyleSheet("font-weight: bold; color: red;")
            elif rage_level_name == "blocking":
                self.state_label.setText("🚫 BLOCKED")
                self.state_label.setStyleSheet("font-weight: bold; color: darkred; font-size: 16px;")
        
        # Update overlay with rage level
        rage_level = RageLevel(rage_level_name)
        self.overlay_manager.update_rage_level(rage_level, rage_fill, target_opacity)
        
        # Update tray
        if self.tray_manager:
            self.tray_manager.set_state(rage_level_name)
    
    def _on_state_changed(self, state_name: str) -> None:
        """Handle state change from audio thread."""
        # This method is now mainly for logging - rage changes handle UI updates
        logger.debug(f"Volume state changed to: {state_name}")
        
        # Note: Overlay updates are now handled by _on_rage_changed() method only
    
    def _on_calibration_sample(self, volume_db: float) -> None:
        """Handle calibration sample."""
        if self.calibration_progress and self.calibration_progress.isVisible():
            current_value = self.calibration_progress.value()
            self.calibration_progress.setValue(current_value + 1)
            
            # Check if calibration is complete
            if current_value >= self.calibration_progress.maximum() - 1:
                # Calibration finished
                if self.calibration_progress:
                    self.calibration_progress.setVisible(False)
                
                if self.calibrate_button:
                    self.calibrate_button.setEnabled(True)
                    self.calibrate_button.setText("🎯 Calibrate (5s)")
                
                # Re-enable start/stop buttons
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(True)
                
                baseline = self.audio_thread.stop_calibration()
                
                # Stop monitoring if it wasn't running before calibration
                if hasattr(self, '_was_monitoring_before_calibration') and not self._was_monitoring_before_calibration:
                    self._stop_monitoring()
                
                # Threshold is saved to config automatically
                
                self.status_bar.showMessage(f"Calibration complete! New threshold: {baseline:.1f} dB", 3000)
                logger.info(f"Voice calibration completed with threshold: {baseline:.1f} dB")
    
    def _update_ui(self) -> None:
        """Update UI elements periodically."""
        if self.audio_thread and self.audio_thread.running:
            stats = self.audio_thread.get_stats()
            
            # Update status text
            status_text = f"FPS: {stats['fps']:.1f} | Frames: {stats['frames_processed']} | State: {stats['audio_state']}"
            if 'capture' in stats:
                status_text += f" | Errors: {stats['capture']['callback_errors']}"
            
            if self.status_text:
                self.status_text.setPlainText(status_text)
    
    def closeEvent(self, event) -> None:
        """Handle window close event."""
        # Always close when X is clicked
        self._stop_monitoring()
        event.accept()
    
    def show_window(self) -> None:
        """Show and raise window."""
        self.show()
        self.raise_()
        self.activateWindow() 