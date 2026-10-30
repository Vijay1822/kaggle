"""
Entry point for the Network Intrusion Detection pipeline.

Usage:
    python main.py

Place train.csv and test.csv in ./data/, the current directory,
or /kaggle/input/... before running.
"""

from src.pipeline import main

if __name__ == "__main__":
    main()
