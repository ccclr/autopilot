"""Controller package exports without eagerly loading trainer dependencies."""

__all__ = ["AutopilotController", "AutobahnController"]


def __getattr__(name):
    if name in __all__:
        from .controller import AutopilotController

        return AutopilotController
    raise AttributeError(name)
