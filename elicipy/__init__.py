"""
EliciPy - A package for elicitation and statistical modeling.

Author: Mattia de' Michieli Vitturi
"""

__version__ = "2.0.0"

from .core import run_elicitation
from .app import run_streamlit

__all__ = ["run_elicitation", "run_streamlit"]

