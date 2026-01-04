import sys
from pathlib import Path

# Add src to sys.path to allow running without installation
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from cli import main

if __name__ == "__main__":
    main()
