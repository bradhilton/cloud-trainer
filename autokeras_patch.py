import autokeras as ak
from autokeras.graph import Graph
import keras
from keras_tuner import HyperParameters
from keras_tuner.engine import hyperparameters
from proxy_auto_model import ProxyAutoModel
from typing import Any, cast, Optional, Union


def apply() -> None:
    ak.AutoModel.__new__ = _AutoModel__new__patch
    ak.AutoModel.__init__ = _AutoModel__init__patch
    Graph._compile_keras_model = _Graph_compile_keras_model_patch


_AutoModel__new__ = ak.AutoModel.__new__
_AutoModel__init__ = ak.AutoModel.__init__
_Graph_compile_keras_model = Graph._compile_keras_model


def _AutoModel__new__patch(
    cls: type[ak.AutoModel],
    *args: Any,
    host: Union[str, tuple[str, int], None] = None,
    **kwargs: Any,
) -> ak.AutoModel:
    if host:
        return cast(
            ak.AutoModel,
            ProxyAutoModel(
                host=host if isinstance(host, str) else f"{host[0]}:{host[1]}",
                args=args,
                kwargs=kwargs,
            ),
        )

    return _AutoModel__new__(cls)


def _AutoModel__init__patch(self: ak.AutoModel, *args: Any, **kwargs: Any) -> None:
    compile_choices = [
        (key, kwargs.pop(key, None)) for key in ("optimizer", "learning_rate")
    ]

    _AutoModel__init__(self, *args, **kwargs)

    setattr(
        self.tuner.hypermodel,
        "_compile_choices",
        {
            name: (
                choice
                if isinstance(choice, hyperparameters.Choice)
                else hyperparameters.Choice(name, [choice])
            )
            for name, choice in compile_choices
            if choice is not None
        },
    )


def _Graph_compile_keras_model_patch(
    self: Graph, hp: HyperParameters, model: keras.Model
) -> keras.Model:
    __Choice = hp.Choice

    def Choice(name: str, values: list, *args: Any, **kwargs: Any) -> Any:
        choice: Optional[hyperparameters.Choice] = getattr(
            self, "_compile_choices", {}
        ).get(name)
        if choice:
            values = choice.values
            kwargs["ordered"] = choice.ordered
            kwargs["default"] = choice.default
        return __Choice(name, values, *args, **kwargs)

    hp.Choice = Choice

    return _Graph_compile_keras_model(self, hp, model)
