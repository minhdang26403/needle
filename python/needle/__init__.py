"""
Needle (Necessary Elements of Deep Learning)

This package provides a minimalist deep learning framework with a learning-first design.
"""

from importlib.metadata import PackageNotFoundError as _PkgNotFoundError
from importlib.metadata import version as _pkg_version

__all__ = ["__version__"]

try:
    __version__ = _pkg_version("needle-dlsys")
except _PkgNotFoundError:
    # Fallback for editable installs before metadata is written
    __version__ = "0.0.1"
