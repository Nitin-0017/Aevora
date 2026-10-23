"""Launch: .venv/bin/python -m streamlit run dashboard.py"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from aevora.dashboard.app import main

main()
