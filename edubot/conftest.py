"""
conftest.py -- pytest configuration for the edubot test suite.
Adds the src/ directory and the dataset scripts/ to sys.path so
test_tools.py can import from tools.py and edubot_context.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str((ROOT / "../EduBot_Dataset_v1.2/scripts").resolve()))
