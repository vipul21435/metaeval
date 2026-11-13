"""Allow ``python -m metaeval``."""

from __future__ import annotations

from metaeval.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
