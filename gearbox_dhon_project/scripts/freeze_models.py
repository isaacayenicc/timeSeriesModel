"""Command wrapper for freeze."""
import sys

from gearbox_dhon.cli import main

if __name__ == "__main__":
    main(["freeze", *sys.argv[1:]])
