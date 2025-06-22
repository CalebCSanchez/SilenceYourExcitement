#!/usr/bin/env python3
"""Setup script for Rage-Reducer."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "sounddevice>=0.4.6",
        "numpy>=1.24.3",
        "PyQt6>=6.6.1",
        "psutil>=5.9.6",
        "onnxruntime>=1.16.3",
        "requests>=2.31.0"
    ]

setup(
    name="rage-reducer",
    version="1.0.0",
    description="A desktop utility to reduce gaming rage through visual feedback",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Rage-Reducer Team",
    author_email="",
    url="https://github.com/yourusername/rage-reducer",
    packages=find_packages(exclude=["tests*"]),
    package_data={
        "rage_reducer": [
            "assets/*",
            "assets/**/*",
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-qt>=4.2.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "ruff>=0.1.6",
            "mypy>=1.7.1",
            "pre-commit>=3.5.0",
            "pyinstaller>=6.0.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.16.3",
        ],
        "audio": [
            "rnnoise>=1.0.2",
            "librosa>=0.10.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "rage-reducer=rage_reducer.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Games/Entertainment",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: Microsoft :: Windows",
        "Environment :: Win32 (MS Windows)",
        "Environment :: X11 Applications :: Qt",
    ],
    keywords="gaming voice control audio overlay rage management",
    python_requires=">=3.10",
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/yourusername/rage-reducer/issues",
        "Source": "https://github.com/yourusername/rage-reducer",
        "Documentation": "https://github.com/yourusername/rage-reducer/wiki",
    },
) 