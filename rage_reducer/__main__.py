#!/usr/bin/env python3
"""
Main entry point for Rage-Reducer application.
This module initializes the GUI and starts all background threads.
"""

import sys
import logging
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

from .ui.settings import MainSettingsWindow
from .core.config import Config


def setup_logging() -> None:
    """Configure logging for the application."""
    log_level = logging.DEBUG if "--debug" in sys.argv else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("rage_reducer.log")
        ]
    )


def main() -> int:
    """Main entry point for the application."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Rage-Reducer v1.0.0")
    
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("Rage-Reducer")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Rage-Reducer Team")
    
    # Set application icon
    icon_path = Path(__file__).parent / "assets" / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = Config()
        
        # Create and show main settings window
        logger.info("Creating main window...")
        main_window = MainSettingsWindow(config)
        logger.info("Showing main window...")
        main_window.show()
        
        logger.info("Application started successfully")
        return app.exec()
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 