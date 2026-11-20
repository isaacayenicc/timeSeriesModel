"""Command wrapper for benchmark."""
import sys

from gearbox_dhon.cli import main

if __name__ == "__main__":
    main(["benchmark", *sys.argv[1:]])
