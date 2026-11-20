"""Command wrapper for audit."""
import sys

from gearbox_dhon.cli import main

if __name__ == "__main__":
    main(["audit", *sys.argv[1:]])
