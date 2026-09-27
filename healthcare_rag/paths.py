"""Shared repository paths, independent of the current working directory."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
