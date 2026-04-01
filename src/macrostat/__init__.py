from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("MacroStat")
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
