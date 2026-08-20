"""Allow ``python -m digital_air_cv``."""

from multiprocessing import freeze_support

from digital_air_cv.cli import main

if __name__ == "__main__":
    freeze_support()
    main()
