"""Command wrapper for run."""
import sys

from gearbox_dhon.cli import main

if __name__ == "__main__":
    main(["run", *sys.argv[1:]])
