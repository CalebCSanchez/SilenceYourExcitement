# 🎮 Rage-Reducer

> Sometimes, we get a little too into games, and our neighbors don't like that. This will help you do just that!

**Rage-Reducer** is a Python-based desktop utility that monitors your microphone input and provides visual feedback when your voice gets too loud during gaming sessions. It overlays a red vignette effect on your screen that intensifies as your volume increases, helping you maintain better voice control and be more considerate to others around you.

## ✨ Features

### 🎯 Core Functionality
- **Real-time voice monitoring** with minimal latency (~20ms)
- **Visual feedback overlay** with smooth animations
- **Voice activity detection** to ignore background noise
- **Noise suppression** using RNNoise to filter out keyboard clicks, fans, etc.
- **Smart calibration** system for personalized volume thresholds

### 🎮 Gaming Integration
- **Game-specific monitoring** - only activate when your target game is in focus
- **Automatic game detection** for popular gaming platforms (Steam, Epic, etc.)
- **Always-on mode** for system-wide monitoring

### 🖥️ User Experience
- **System tray integration** with status indicators
- **Modern PyQt6 interface** with playful design
- **Configurable thresholds** and visual settings
- **Windows startup integration**
- **Multi-monitor support**

### 📊 Visual Feedback States
- **😌 Calm** - Normal voice levels, no overlay
- **😐 Warning** - Elevated voice, partial red vignette from edges
- **😠 Angry** - Loud voice, full screen red overlay (flashlight-blind effect)

## 🚀 Quick Start

### Prerequisites
- Windows 10/11 (x64)
- Python 3.10+ (for development)
- Microphone/audio input device

### Installation

#### Option 1: Download Executable (Recommended)
1. Download the latest `RageReducer.exe` from [Releases](https://github.com/yourusername/rage-reducer/releases)
2. Run the executable
3. Follow the setup wizard

#### Option 2: Install from Source
```bash
# Clone the repository
git clone https://github.com/yourusername/rage-reducer.git
cd rage-reducer

# Set up development environment
make dev

# Run the application
make run
```

### First-Time Setup

1. **Select your microphone** from the dropdown
2. **Calibrate your voice**:
   - Click "🎯 Calibrate (5s)"
   - Speak normally for 5 seconds
   - The system will learn your baseline volume
3. **Choose target game** (optional):
   - Select "[Always active]" for system-wide monitoring
   - Or pick a specific game from the detected list
4. **Click "▶️ Start Monitoring"**

## 🎮 How It Works

### Audio Pipeline
```
Microphone → Noise Suppression → Voice Activity Detection → Volume Analysis → State Machine → Visual Overlay
```

1. **Audio Capture**: Records 16kHz mono audio in ~20ms chunks
2. **Noise Suppression**: RNNoise removes background noise
3. **Voice Activity Detection**: Silero-VAD identifies speech vs. silence
4. **Volume Analysis**: Calculates RMS volume in dBFS
5. **State Machine**: Applies debouncing and threshold logic
6. **Visual Feedback**: Updates screen overlay with smooth animations

### Volume Thresholds
- **Baseline**: Your normal speaking volume (auto-calibrated)
- **Warning**: Baseline + 3dB (partial overlay)
- **Angry**: Baseline + 6dB (full overlay)

### Debouncing Logic
- **Warning state**: Immediate activation
- **Angry state**: Requires 3 consecutive loud frames to prevent false triggers
- **Return to calm**: Immediate when volume drops

## 📁 Project Structure

```
rage_reducer/
├── __init__.py          # Package initialization
├── __main__.py          # Application entry point
├── core/                # Core functionality
│   ├── state.py         # Volume state management & debouncing
│   ├── config.py        # Configuration management
│   └── gamecheck.py     # Game process detection
├── audio/               # Audio processing pipeline
│   ├── capture.py       # Real-time audio capture
│   ├── suppress.py      # Noise suppression (RNNoise)
│   └── vad.py          # Voice activity detection (Silero-VAD)
├── ui/                  # User interface components
│   ├── overlay.py       # Screen overlay with vignette effect
│   ├── settings.py      # Main settings window
│   └── tray.py         # System tray integration
├── assets/              # Icons and resources
tests/                   # Test suite
Makefile                 # Development commands
setup.py                 # Package configuration
requirements.txt         # Dependencies
```

## 🛠️ Development

### Setup Development Environment
```bash
# Clone and setup
git clone https://github.com/yourusername/rage-reducer.git
cd rage-reducer
make dev

# Activate virtual environment
venv/Scripts/activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### Common Commands
```bash
make help          # Show all available commands
make run           # Run the application
make run-debug     # Run with debug logging
make test          # Run tests with coverage
make lint          # Run code quality checks
make format        # Format code with black/isort
make build         # Build executable with PyInstaller
make clean         # Clean build artifacts
```

### Code Quality
The project uses several tools to maintain code quality:
- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Fast linting
- **MyPy**: Type checking
- **Pytest**: Testing framework
- **Pre-commit**: Git hooks for automatic checks

### Testing
```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_state.py -v

# Run with coverage report
python -m pytest --cov=rage_reducer --cov-report=html
```

## 🔧 Configuration

### Configuration File
Settings are stored in `~/.rage_reducer/config.json`:

```json
{
  "audio": {
    "device_index": 0,
    "device_name": "Default Microphone",
    "sample_rate": 16000,
    "baseline_volume": -28.5,
    "warning_threshold": -25.5,
    "angry_threshold": -22.5
  },
  "overlay": {
    "animation_duration": 500,
    "warning_opacity": 0.4,
    "angry_opacity": 0.85
  },
  "game": {
    "selected_game": "[Always active]",
    "enabled": false
  },
  "app": {
    "start_with_windows": false,
    "minimize_to_tray": true
  }
}
```

### Environment Variables (Development)
```bash
# Copy and modify for your environment
cp .env.example .env

# Key variables:
LOG_LEVEL=DEBUG
ENABLE_NOISE_SUPPRESSION=true
ENABLE_VAD=true
```

## 🎯 Advanced Usage

### Command Line Options
```bash
# Run with debug logging
python -m rage_reducer --debug

# Specify custom config file
python -m rage_reducer --config /path/to/config.json
```

### Game Detection
The application can automatically detect running games by analyzing:
- Process names and window titles
- Installation paths (Steam, Epic Games, etc.)
- Graphics API usage (DirectX, Vulkan)
- Game engine signatures (Unity, Unreal)

### Multi-Monitor Support
Overlays automatically appear on all connected monitors. Each monitor gets its own overlay instance with synchronized animations.

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines
- Follow the existing code style (enforced by pre-commit hooks)
- Add tests for new functionality
- Update documentation for user-facing changes
- Keep commits atomic and well-described

## 📋 Requirements

### Runtime Dependencies
- **Python 3.10+** (for development)
- **Windows 10/11** (primary target platform)
- **Audio input device** (microphone)

### Key Libraries
- **PyQt6** - Modern GUI framework
- **sounddevice** - Real-time audio I/O
- **numpy** - Numerical operations
- **onnxruntime** - Silero-VAD inference
- **psutil** - System process monitoring
- **pywin32** - Windows API access

### Optional Dependencies
- **rnnoise** - Advanced noise suppression
- **onnxruntime-gpu** - GPU-accelerated inference
- **librosa** - Advanced audio processing

## 🐛 Troubleshooting

### Common Issues

#### No Audio Devices Found
- Ensure your microphone is connected and enabled
- Check Windows audio settings
- Try running as administrator

#### Overlay Not Appearing
- Check that the application is running (system tray icon)
- Verify game detection is working
- Try "[Always active]" mode

#### High CPU Usage
- Disable noise suppression in settings
- Reduce audio sample rate
- Check for audio driver issues

#### Game Not Detected
- Click "🔄 Refresh" to rescan for games
- Use "[Always active]" mode as fallback
- Check that the game window has focus

### Debug Mode
Run with debug logging for detailed troubleshooting:
```bash
python -m rage_reducer --debug
```

Check the log file: `rage_reducer.log`

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Silero Team** - For the excellent VAD model
- **RNNoise** - For noise suppression capabilities
- **Qt/PyQt** - For the robust GUI framework
- **Gaming Community** - For inspiration and feedback

## 🔗 Links

- **Documentation**: [Wiki](https://github.com/yourusername/rage-reducer/wiki)
- **Bug Reports**: [Issues](https://github.com/yourusername/rage-reducer/issues)
- **Feature Requests**: [Discussions](https://github.com/yourusername/rage-reducer/discussions)
- **Releases**: [Latest Version](https://github.com/yourusername/rage-reducer/releases)

---

**Made with ❤️ for the gaming community**

*Remember: The goal isn't to silence yourself, but to be more aware of your voice levels and considerate to those around you. Game responsibly!* 🎮
