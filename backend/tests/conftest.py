"""
Pytest configuration for backend tests.
Adds backend root to sys.path so we can import from ml/, app/ directories.
"""
import sys
from pathlib import Path

# Add backend root FIRST so backend/ml/ takes precedence over project-root/ml/
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))