"""InstaGeo New Apps package."""
import importlib


def __getattr__(name):
    if name == "backend":
        mod = importlib.import_module(__name__ + ".backend")
        globals()["backend"] = mod
        return mod
    raise AttributeError(name)
