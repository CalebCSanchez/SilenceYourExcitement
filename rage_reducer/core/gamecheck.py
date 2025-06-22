"""
Game process detection and foreground window monitoring.
Used to enable game-specific rage reduction functionality.
"""

import logging
import time
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import psutil

# Windows-specific imports
try:
    import win32gui
    import win32process
    import win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    logging.warning("pywin32 not available - game detection disabled")

logger = logging.getLogger(__name__)


@dataclass
class GameProcess:
    """Represents a detected game process."""
    pid: int
    name: str
    executable: str
    window_title: str
    is_foreground: bool = False


class GameChecker:
    """Monitors running games and foreground window status."""
    
    def __init__(self):
        """Initialize game checker."""
        self.last_scan_time = 0.0
        self.scan_interval = 2.0  # Scan for new games every 2 seconds
        self.foreground_check_interval = 0.5  # Check foreground every 500ms
        self.last_foreground_check = 0.0
        
        # Cache
        self._detected_games: Dict[int, GameProcess] = {}
        self._current_foreground_pid: Optional[int] = None
        self._current_foreground_title: str = ""
        
        # Game detection patterns
        self._game_keywords = {
            "steam", "game", "play", "dx", "vulkan", "opengl", "unity",
            "unreal", "engine", "fps", "rpg", "mmo", "battle", "war",
            "craft", "simulator", "racing", "sport", "adventure"
        }
        
        # Common non-game processes to ignore
        self._ignore_processes = {
            "explorer.exe", "winlogon.exe", "csrss.exe", "wininit.exe",
            "services.exe", "lsass.exe", "svchost.exe", "dwm.exe",
            "chrome.exe", "firefox.exe", "edge.exe", "notepad.exe",
            "code.exe", "devenv.exe", "python.exe", "pythonw.exe"
        }
    
    def get_available_games(self) -> List[GameProcess]:
        """
        Get list of currently running games.
        
        Returns:
            List of detected game processes
        """
        current_time = time.time()
        
        # Only scan periodically to avoid performance issues
        if current_time - self.last_scan_time < self.scan_interval:
            return list(self._detected_games.values())
        
        self.last_scan_time = current_time
        
        if not HAS_WIN32:
            return []
        
        # Clear old games
        self._detected_games.clear()
        
        try:
            # Get all processes
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    proc_info = proc.info
                    if not proc_info['exe'] or not proc_info['name']:
                        continue
                    
                    # Skip system processes
                    if proc_info['name'].lower() in self._ignore_processes:
                        continue
                    
                    # Get window title if process has windows
                    window_titles = self._get_process_windows(proc_info['pid'])
                    if not window_titles:
                        continue
                    
                    main_title = window_titles[0]  # Use first/main window title
                    
                    # Check if this looks like a game
                    if self._is_likely_game(proc_info['name'], main_title, proc_info['exe']):
                        game = GameProcess(
                            pid=proc_info['pid'],
                            name=proc_info['name'],
                            executable=proc_info['exe'],
                            window_title=main_title
                        )
                        self._detected_games[proc_info['pid']] = game
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    continue
        
        except Exception as e:
            logger.error(f"Error scanning for games: {e}")
        
        logger.debug(f"Found {len(self._detected_games)} potential games")
        return list(self._detected_games.values())
    
    def is_game_foreground(self, game_name: str) -> bool:
        """
        Check if specified game is currently in foreground.
        
        Args:
            game_name: Name of the game to check
            
        Returns:
            True if game is in foreground
        """
        if not HAS_WIN32 or game_name == "[Always active]":
            return True
        
        current_time = time.time()
        
        # Only check periodically to avoid performance issues
        if current_time - self.last_foreground_check < self.foreground_check_interval:
            # Use cached result
            return self._is_cached_game_foreground(game_name)
        
        self.last_foreground_check = current_time
        
        try:
            # Get foreground window
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return False
            
            # Get window title
            window_title = win32gui.GetWindowText(hwnd)
            self._current_foreground_title = window_title
            
            # Get process ID
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            self._current_foreground_pid = pid
            
            # Check if this matches our target game
            if pid in self._detected_games:
                game = self._detected_games[pid]
                is_match = (
                    game.name.lower() == game_name.lower() or
                    game_name.lower() in game.window_title.lower() or
                    game_name.lower() in window_title.lower()
                )
                return is_match
            
            # Check by name/title matching
            return (
                game_name.lower() in window_title.lower() or
                any(game_name.lower() == game.name.lower() 
                    for game in self._detected_games.values())
            )
            
        except Exception as e:
            logger.error(f"Error checking foreground game: {e}")
            return False
    
    def get_foreground_game(self) -> Optional[GameProcess]:
        """
        Get currently foreground game process.
        
        Returns:
            GameProcess if foreground window is a detected game, None otherwise
        """
        if not HAS_WIN32:
            return None
        
        if self._current_foreground_pid and self._current_foreground_pid in self._detected_games:
            game = self._detected_games[self._current_foreground_pid]
            game.is_foreground = True
            return game
        
        return None
    
    def _get_process_windows(self, pid: int) -> List[str]:
        """Get window titles for a process."""
        if not HAS_WIN32:
            return []
        
        window_titles = []
        
        def enum_windows_callback(hwnd, _):
            try:
                _, window_pid = win32process.GetWindowThreadProcessId(hwnd)
                if window_pid == pid:
                    # Only get visible windows with titles
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        if title.strip():
                            window_titles.append(title)
            except:
                pass
            return True
        
        try:
            win32gui.EnumWindows(enum_windows_callback, None)
        except:
            pass
        
        return window_titles
    
    def _is_likely_game(self, process_name: str, window_title: str, executable_path: str) -> bool:
        """
        Heuristic to determine if a process is likely a game.
        
        Args:
            process_name: Process executable name
            window_title: Main window title
            executable_path: Full path to executable
            
        Returns:
            True if process appears to be a game
        """
        # Convert to lowercase for comparison
        name_lower = process_name.lower()
        title_lower = window_title.lower()
        path_lower = executable_path.lower() if executable_path else ""
        
        # Skip obvious non-games
        if any(ignore in name_lower for ignore in self._ignore_processes):
            return False
        
        # Check for game-related keywords
        all_text = f"{name_lower} {title_lower} {path_lower}"
        
        # Common game indicators
        game_indicators = [
            # Common game directories
            "steam" in path_lower and "steamapps" in path_lower,
            "program files" in path_lower and any(kw in all_text for kw in ["game", "play"]),
            "epic games" in path_lower,
            "origin games" in path_lower,
            "ubisoft" in path_lower,
            
            # Common game engines/frameworks
            any(engine in all_text for engine in ["unity", "unreal", "source", "frostbite"]),
            
            # Graphics API indicators
            any(gfx in all_text for gfx in ["dx11", "dx12", "directx", "vulkan", "opengl"]),
            
            # Game genre keywords
            any(genre in all_text for genre in [
                "fps", "rpg", "mmo", "rts", "moba", "battle", "war", "craft",
                "simulator", "racing", "sport", "adventure", "puzzle", "strategy"
            ]),
            
            # Has fullscreen window (common for games)
            self._has_fullscreen_window(window_title),
        ]
        
        # Must have at least one positive indicator
        if not any(game_indicators):
            return False
        
        # Additional check: process should have meaningful window
        if not window_title or len(window_title.strip()) < 3:
            return False
        
        logger.debug(f"Detected potential game: {process_name} - {window_title}")
        return True
    
    def _has_fullscreen_window(self, window_title: str) -> bool:
        """Check if process has fullscreen windows (game indicator)."""
        # This is a placeholder - could be enhanced to actually check window size
        # For now, just check for common fullscreen game title patterns
        fullscreen_patterns = ["fullscreen", "borderless", "exclusive"]
        return any(pattern in window_title.lower() for pattern in fullscreen_patterns)
    
    def _is_cached_game_foreground(self, game_name: str) -> bool:
        """Use cached foreground information to check game status."""
        if not self._current_foreground_title:
            return False
        
        return (
            game_name.lower() in self._current_foreground_title.lower() or
            any(game_name.lower() == game.name.lower() and game.pid == self._current_foreground_pid
                for game in self._detected_games.values())
        )
    
    def refresh_games(self) -> List[GameProcess]:
        """Force refresh of game list."""
        self.last_scan_time = 0.0  # Force rescan
        return self.get_available_games()
    
    def get_game_names(self) -> List[str]:
        """Get list of game names for UI dropdown."""
        games = self.get_available_games()
        names = ["[Always active]"]  # Default option
        names.extend(game.name for game in games)
        return names 