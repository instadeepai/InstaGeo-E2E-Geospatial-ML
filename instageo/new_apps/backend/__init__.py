"""InstaGeo Backend package."""
import importlib


def __getattr__(name):
    if name == "app":
        mod = importlib.import_module(__name__ + ".app")
        globals()["app"] = mod
        return mod
    raise AttributeError(name)
