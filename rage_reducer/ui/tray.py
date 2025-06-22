"""
System tray manager for Rage-Reducer.
Provides system tray icon and context menu.
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import QSystemTrayIcon, QMenu, QApplication
from PyQt6.QtGui import QIcon, QPixmap, QAction
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


class TrayManager:
    """Manages system tray icon and functionality."""
    
    def __init__(self, main_window):
        """
        Initialize tray manager.
        
        Args:
            main_window: Reference to main window
        """
        self.main_window = main_window
        self.tray_icon: Optional[QSystemTrayIcon] = None
        self.current_state = "idle"
        
        self._setup_tray()
    
    def _setup_tray(self) -> None:
        """Setup system tray icon and menu."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("System tray not available")
            return
        
        # Create tray icon
        self.tray_icon = QSystemTrayIcon()
        
        # Set initial icon
        self._update_icon("idle")
        
        # Create context menu
        menu = QMenu()
        
        # Show/Hide window action
        show_action = QAction("Show Window", menu)
        show_action.triggered.connect(self.main_window.show_window)
        menu.addAction(show_action)
        
        menu.addSeparator()
        
        # Start/Stop monitoring actions (placeholder)
        self.start_action = QAction("Start Monitoring", menu)
        self.start_action.triggered.connect(self._toggle_monitoring)
        menu.addAction(self.start_action)
        
        menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", menu)
        exit_action.triggered.connect(self._exit_application)
        menu.addAction(exit_action)
        
        # Set context menu
        self.tray_icon.setContextMenu(menu)
        
        # Connect double-click to show window
        self.tray_icon.activated.connect(self._on_tray_activated)
        
        # Set tooltip
        self.tray_icon.setToolTip("Rage-Reducer - Gaming Voice Control")
    
    def _update_icon(self, state: str) -> None:
        """Update tray icon based on state."""
        if not self.tray_icon:
            return
        
        # Create colored icon based on state
        pixmap = QPixmap(32, 32)
        
        if state == "idle":
            pixmap.fill(Qt.GlobalColor.gray)
        elif state == "monitoring":
            pixmap.fill(Qt.GlobalColor.green)
        elif state == "calm":
            pixmap.fill(Qt.GlobalColor.green)
        elif state == "warning":
            pixmap.fill(Qt.GlobalColor.yellow)
        elif state == "angry":
            pixmap.fill(Qt.GlobalColor.red)
        else:
            pixmap.fill(Qt.GlobalColor.gray)
        
        icon = QIcon(pixmap)
        self.tray_icon.setIcon(icon)
    
    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Handle tray icon activation."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.main_window.show_window()
    
    def _toggle_monitoring(self) -> None:
        """Toggle monitoring state."""
        # This would connect to the main window's start/stop functionality
        # For now, just show the window
        self.main_window.show_window()
    
    def _exit_application(self) -> None:
        """Exit the application."""
        QApplication.quit()
    
    def set_state(self, state: str) -> None:
        """
        Set current state and update icon.
        
        Args:
            state: Current state (idle, monitoring, calm, warning, angry)
        """
        if state != self.current_state:
            self.current_state = state
            self._update_icon(state)
            
            # Update tooltip
            state_text = {
                "idle": "Idle",
                "monitoring": "Monitoring",
                "calm": "Calm 😌",
                "warning": "Warning 😐",
                "angry": "Angry 😠"
            }.get(state, "Unknown")
            
            if self.tray_icon:
                self.tray_icon.setToolTip(f"Rage-Reducer - {state_text}")
    
    def show(self) -> None:
        """Show tray icon."""
        if self.tray_icon:
            self.tray_icon.show()
    
    def hide(self) -> None:
        """Hide tray icon."""
        if self.tray_icon:
            self.tray_icon.hide()
    
    def isVisible(self) -> bool:
        """Check if tray icon is visible."""
        return self.tray_icon.isVisible() if self.tray_icon else False
    
    def showMessage(self, title: str, message: str, icon: QSystemTrayIcon.MessageIcon, timeout: int = 5000) -> None:
        """Show tray notification."""
        if self.tray_icon:
            self.tray_icon.showMessage(title, message, icon, timeout) 