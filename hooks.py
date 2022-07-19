from typing import Callable


class Hook:
    def __init__(self, method: Callable, hook: Callable) -> None:
        self.method = method
        self.hook = hook

    def __enter__(self) -> None:
        self._hooks().add(self.hook)

    def __exit__(self, exc_type, exc_value, tb) -> None:
        self._hooks().remove(self.hook)

    def _hooks(self) -> set[Callable]:
        owner = self.method.__self__
        name = self.method.__name__

        if not hasattr(owner, "__hooks"):
            setattr(owner, "__hooks", {})

        all_hooks: dict[str, set[Callable]] = getattr(owner, "__hooks")

        if name not in all_hooks:

            def patch(*args, **kwargs):
                for hook in self._hooks():
                    hook(*args, **kwargs)
                return self.method(*args, **kwargs)

            setattr(patch, "__self__", owner)
            setattr(patch, "__name__", name)
            setattr(owner, name, patch)
            all_hooks[name] = set()

        return all_hooks[name]
