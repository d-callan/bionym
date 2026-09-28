"""bionym: bioinformatics ID -> confidence-scored knowledge graph via JEV."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bionym")
except PackageNotFoundError:
    # source tree without an install (tests set their own expectations)
    __version__ = "0.0.0.dev0"
