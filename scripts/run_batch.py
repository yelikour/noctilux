from __future__ import annotations

import sys

from noctilux.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
