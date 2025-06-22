"""
Screen overlay UI with red vignette effect.
Provides visual feedback based on rage level system.
"""

import logging
import math
import time
from typing import Optional

from PyQt6.QtWidgets import QWidget, QApplication, QGraphicsOpacityEffect
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, QRect
from PyQt6.QtGui import QPainter, QColor, QBrush, QRadialGradient, QScreen

from ..core.state import VolumeState, RageLevel

logger = logging.getLogger(__name__)


class OverlayWidget(QWidget):
    """Full-screen overlay widget with animated red vignette effect based on rage level."""
    
    def __init__(self, screen: Optional[QScreen] = None):
        """
        Initialize overlay widget.
        
        Args:
            screen: Target screen (None for primary screen)
        """
        super().__init__()
        
        self.target_screen = screen or QApplication.primaryScreen()
        self.current_rage_level = RageLevel.CALM
        self.current_rage_fill = 0.0
        
        # Animation properties
        self._opacity = 0.0
        
        # Visual configuration
        self.animation_duration = 300  # ms - faster transitions for rage system
        self.update_interval = 16  # ms - 60fps updates
        
        # Colors
        self.vignette_color = QColor(255, 0, 0)  # Red
        
        # Animations
        self.opacity_animation: Optional[QPropertyAnimation] = None
        
        # Update timer for smooth rage level updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        
        self._setup_window()
        self._setup_animations()
    
    def _setup_window(self) -> None:
        """Configure window properties."""
        # Make window frameless and transparent
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.BypassWindowManagerHint
        )
        
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        # Additional attributes to ensure mouse transparency
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        
        # Set window geometry to cover entire screen
        screen_geometry = self.target_screen.geometry()
        self.setGeometry(screen_geometry)
        
        # Initially hidden
        self.hide()
        
        logger.info(f"Overlay window setup - Screen: {screen_geometry}")
    
    def _setup_animations(self) -> None:
        """Setup property animations."""
        # Opacity animation
        self.opacity_animation = QPropertyAnimation(self, b"opacity")
        self.opacity_animation.setDuration(self.animation_duration)
        self.opacity_animation.setEasingCurve(QEasingCurve.Type.OutQuad)
    
    @pyqtProperty(float)
    def opacity(self) -> float:
        """Get current vignette opacity."""
        return self._opacity
    
    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set vignette opacity."""
        self._opacity = value
        self.update()
    
    def update_rage_level(self, rage_level: RageLevel, rage_fill: float, target_opacity: float) -> None:
        """
        Update overlay based on rage level and fill amount.
        
        Args:
            rage_level: Current rage level category
            rage_fill: Current rage meter fill (0.0 to 3.0)
            target_opacity: Target opacity for the overlay (0.0 to 1.0)
        """
        # Prevent rapid updates - only update if there's a significant change
        current_time = time.time()
        if not hasattr(self, '_last_update_time'):
            self._last_update_time = 0
        
        # Check if this is a significant change worth updating
        significant_change = (
            rage_level != self.current_rage_level or  # Level changed
            abs(target_opacity - self._opacity) > 0.1 or  # Opacity changed significantly
            current_time - self._last_update_time > 1.0  # Been a while since last update
        )
        
        if not significant_change:
            return  # Skip minor updates to prevent flashing
        
        logger.info(f"Overlay rage update: {rage_level.value}, fill: {rage_fill:.2f}, opacity: {target_opacity:.2f}")
        self._last_update_time = current_time
        
        self.current_rage_level = rage_level
        self.current_rage_fill = rage_fill
        
        # Show/hide overlay based on rage level
        if target_opacity > 0.0:
            self._ensure_visible()
            self._animate_to_opacity(target_opacity)
        else:
            self._animate_to_opacity(0.0)
    
    def update_state(self, state: VolumeState) -> None:
        """
        Legacy method for compatibility - now uses simple state mapping.
        
        Args:
            state: Current volume state
        """
        # Map simple states to rage levels for backward compatibility
        if state == VolumeState.CALM:
            self.update_rage_level(RageLevel.CALM, 0.0, 0.0)
        elif state == VolumeState.WARNING:
            self.update_rage_level(RageLevel.MEDIUM, 0.5, 0.2)
        elif state == VolumeState.ANGRY:
            self.update_rage_level(RageLevel.ANGRY, 1.0, 0.6)
    
    def _animate_to_opacity(self, target_opacity: float) -> None:
        """Animate to target opacity."""
        if target_opacity == 0.0:
            # Hide after animation
            self.opacity_animation.finished.connect(self._hide_if_transparent)
        
        # Stop any running animation
        if self.opacity_animation.state() == QPropertyAnimation.State.Running:
            self.opacity_animation.stop()
        
        # Set animation start/end values
        self.opacity_animation.setStartValue(self._opacity)
        self.opacity_animation.setEndValue(target_opacity)
        
        # Start animation
        self.opacity_animation.start()
    
    def _ensure_visible(self) -> None:
        """Ensure overlay is visible."""
        if not self.isVisible():
            logger.info("Making overlay visible")
            self.show()
            self.raise_()
            # Don't activate the window to avoid stealing focus and interfering with mouse events
    
    def _hide_if_transparent(self) -> None:
        """Hide overlay if opacity is zero."""
        if self._opacity <= 0.01:
            self.hide()
        
        # Disconnect the signal to avoid multiple calls
        try:
            self.opacity_animation.finished.disconnect(self._hide_if_transparent)
        except TypeError:
            pass  # Signal not connected

    def _calculate_vignette_radius(self) -> float:
        """Calculate vignette radius based on rage level."""
        screen_size = self.target_screen.size()
        
        if self.current_rage_level == RageLevel.BLOCKING:
            # Full screen coverage for blocking level
            diagonal = math.sqrt(screen_size.width() ** 2 + screen_size.height() ** 2)
            return diagonal
        elif self.current_rage_level == RageLevel.ANGRY:
            # Large coverage for angry level (scales with fill)
            diagonal = math.sqrt(screen_size.width() ** 2 + screen_size.height() ** 2)
            # Scale radius between 70% and 100% of diagonal based on rage fill (1.0 to 3.0)
            fill_progress = (self.current_rage_fill - 1.0) / 2.0  # 0.0 to 1.0
            return diagonal * (0.7 + fill_progress * 0.3)
        elif self.current_rage_level == RageLevel.MEDIUM:
            # Medium coverage for medium level (scales with fill)
            diagonal = math.sqrt(screen_size.width() ** 2 + screen_size.height() ** 2)
            # Scale radius between 40% and 70% of diagonal based on rage fill (0.5 to 1.0)
            fill_progress = (self.current_rage_fill - 0.5) / 0.5  # 0.0 to 1.0
            return diagonal * (0.4 + fill_progress * 0.3)
        else:
            # Small coverage for calm level (fading out)
            diagonal = math.sqrt(screen_size.width() ** 2 + screen_size.height() ** 2)
            # Scale radius based on remaining rage fill (0.0 to 0.5)
            if self.current_rage_fill > 0.0:
                fill_progress = self.current_rage_fill / 0.5  # 0.0 to 1.0
                return diagonal * (0.2 * fill_progress)
            return 0.0

    def paintEvent(self, event) -> None:
        """Paint the vignette effect."""
        if self._opacity <= 0.01:  # Skip painting if nearly transparent
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget dimensions
        rect = self.rect()
        
        if self.current_rage_level == RageLevel.BLOCKING:
            # At blocking level, fill the entire screen with solid red
            blocking_color = QColor(self.vignette_color)
            blocking_color.setAlphaF(self._opacity)
            
            painter.setBrush(QBrush(blocking_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(rect)
            
        else:
            # For other levels, use vignette effect
            center_x = rect.width() / 2
            center_y = rect.height() / 2
            
            # Create radial gradient for vignette effect
            vignette_radius = self._calculate_vignette_radius()
            gradient = QRadialGradient(center_x, center_y, vignette_radius)
            
            # Transparent center to opaque edges
            transparent_color = QColor(self.vignette_color)
            transparent_color.setAlphaF(0.0)
            
            opaque_color = QColor(self.vignette_color)
            opaque_color.setAlphaF(self._opacity)
            
            # Adjust gradient stops based on rage level for more coverage
            if self.current_rage_level == RageLevel.ANGRY:
                # More aggressive vignette for angry level
                gradient.setColorAt(0.0, transparent_color)
                gradient.setColorAt(0.3, transparent_color)  # Smaller clear center
                gradient.setColorAt(1.0, opaque_color)
            elif self.current_rage_level == RageLevel.MEDIUM:
                # Medium vignette for warning level
                gradient.setColorAt(0.0, transparent_color)
                gradient.setColorAt(0.5, transparent_color)  # Medium clear center
                gradient.setColorAt(1.0, opaque_color)
            else:
                # Gentle vignette for calm level
                gradient.setColorAt(0.0, transparent_color)
                gradient.setColorAt(0.7, transparent_color)  # Large clear center
                gradient.setColorAt(1.0, opaque_color)
            
            # Fill the widget with gradient
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(rect)
    
    def set_animation_duration(self, duration_ms: int) -> None:
        """Set animation duration."""
        self.animation_duration = duration_ms
        self.opacity_animation.setDuration(duration_ms)
    
    def set_vignette_color(self, color: QColor) -> None:
        """Set vignette color."""
        self.vignette_color = color
        self.update()
    
    def force_hide(self) -> None:
        """Force hide overlay immediately."""
        # Stop animations
        if self.opacity_animation.state() == QPropertyAnimation.State.Running:
            self.opacity_animation.stop()
        
        # Reset properties
        self._opacity = 0.0
        self.current_rage_level = RageLevel.CALM
        self.current_rage_fill = 0.0
        
        # Hide window
        self.hide()
    
    def get_stats(self) -> dict:
        """Get overlay statistics."""
        return {
            'visible': self.isVisible(),
            'current_state': self.current_rage_level.value,
            'fill': self.current_rage_fill,
            'opacity': self._opacity,
            'animation_duration': self.animation_duration,
            'screen_geometry': {
                'width': self.target_screen.size().width(),
                'height': self.target_screen.size().height(),
                'x': self.target_screen.geometry().x(),
                'y': self.target_screen.geometry().y()
            }
        }


class MultiScreenOverlayManager:
    """Manages overlays across multiple screens."""
    
    def __init__(self, primary_only: bool = True):
        """
        Initialize multi-screen overlay manager.
        
        Args:
            primary_only: If True, only create overlay on primary screen
        """
        self.overlays: dict[QScreen, OverlayWidget] = {}
        self.current_rage_level = RageLevel.CALM
        self.current_rage_fill = 0.0
        self.primary_only = primary_only
        
        # Initialize overlays for screens
        self._create_overlays()
    
    def _create_overlays(self) -> None:
        """Create overlay widgets for selected screens."""
        app = QApplication.instance()
        if not app:
            logger.error("No QApplication instance found")
            return
        
        if self.primary_only:
            # Only create overlay for primary screen
            primary_screen = app.primaryScreen()
            if primary_screen:
                overlay = OverlayWidget(primary_screen)
                self.overlays[primary_screen] = overlay
                logger.info(f"Created overlay for primary screen: {primary_screen.name()}")
        else:
            # Create overlays for all screens
            for screen in app.screens():
                overlay = OverlayWidget(screen)
                self.overlays[screen] = overlay
                logger.info(f"Created overlay for screen: {screen.name()}")
    
    def update_rage_level(self, rage_level: RageLevel, rage_fill: float, target_opacity: float) -> None:
        """Update all overlays with new rage level."""
        self.current_rage_level = rage_level
        self.current_rage_fill = rage_fill
        
        for overlay in self.overlays.values():
            overlay.update_rage_level(rage_level, rage_fill, target_opacity)
    
    def update_state(self, state: VolumeState) -> None:
        """Update all overlays with new state."""
        if state == VolumeState.CALM:
            self.update_rage_level(RageLevel.CALM, 0.0, 0.0)
        elif state == VolumeState.WARNING:
            self.update_rage_level(RageLevel.MEDIUM, 0.5, 0.2)
        elif state == VolumeState.ANGRY:
            self.update_rage_level(RageLevel.ANGRY, 1.0, 0.6)
    
    def force_hide_all(self) -> None:
        """Force hide all overlays."""
        for overlay in self.overlays.values():
            overlay.force_hide()
    
    def set_animation_duration(self, duration_ms: int) -> None:
        """Set animation duration for all overlays."""
        for overlay in self.overlays.values():
            overlay.set_animation_duration(duration_ms)
    
    def set_vignette_color(self, color: QColor) -> None:
        """Set vignette color for all overlays."""
        for overlay in self.overlays.values():
            overlay.set_vignette_color(color)
    
    def get_stats(self) -> dict:
        """Get statistics for all overlays."""
        stats = {
            'current_state': self.current_rage_level.value,
            'fill': self.current_rage_fill,
            'screen_count': len(self.overlays),
            'overlays': {}
        }
        
        for screen, overlay in self.overlays.items():
            stats['overlays'][screen.name()] = overlay.get_stats()
        
        return stats 