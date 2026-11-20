"""Command wrapper for blind."""
import sys

from gearbox_dhon.cli import main

if __name__ == "__main__":
    main(["blind", *sys.argv[1:]])
