"""
WealthDesk package -- SFinal: Session 1 + Session 2 merged
============================================================

This file runs automatically when Python imports the wealthdesk package.
Use it to set up the environment before any other module loads.
"""
import os

os.environ.setdefault("HF_HUB_VERBOSITY", "error")

from dotenv import load_dotenv

load_dotenv()
