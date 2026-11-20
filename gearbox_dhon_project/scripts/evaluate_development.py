"""Command wrapper for analyze."""
import sys

from gearbox_dhon.cli import main

if __name__ == "__main__":
    main(["analyze", *sys.argv[1:]])
