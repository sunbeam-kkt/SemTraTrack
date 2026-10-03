#!/usr/bin/env python3
"""Run the paper-alignment unit tests shipped with this repository."""
import subprocess, sys

if __name__ == '__main__':
    raise SystemExit(subprocess.call([sys.executable, '-m', 'pytest', '-q', 'tests']))
