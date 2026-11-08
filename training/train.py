"""
Main Training Entrypoint for AEN-PPO, Masked PPO, and Standard PPO.
Dispatches training runs with CLI arguments.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.train_visual import main

if __name__ == "__main__":
    main()
