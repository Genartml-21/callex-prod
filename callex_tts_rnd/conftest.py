"""Pytest configuration — add src/ to path for test imports."""
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = str(Path(__file__).parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
