"""Allow ``python -m nike_detection``."""

from multiprocessing import freeze_support

from nike_detection.cli import main

if __name__ == "__main__":
    freeze_support()
    main()
