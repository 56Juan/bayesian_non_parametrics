"""
model_psbp_fd
=============
PSBP extendido a datos funcionales (Chung & Dunson, 2009 → Series de tiempo funcionales).
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("model_psbp_fd")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
