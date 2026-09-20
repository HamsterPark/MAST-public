"""Entry point: `python -m mast` → mast.pipeline.main:main()."""
from mast.pipeline.main import main
import sys

if __name__ == "__main__":
    sys.exit(main())
