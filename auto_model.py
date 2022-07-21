import autokeras as ak
from autokeras.graph import Graph
from dataclasses import dataclass
import httpx
import inspect
from itertools import chain
from keras_tuner.engine import hyperparameters
import pickle
import sys
from typing import Any, cast, Optional, Union


class AutoModel(ak.AutoModel):
    _remote_state: Optional["RemoteState"] = None

    @dataclass
    class RemoteState:
        host: str
        project_name: str

    def __getattribute__(self, __name: str) -> Any:
        if (
            __name == "_remote_state"
            or not self._remote_state
            or __name.startswith("__")
        ):
            return super().__getattribute__(__name)
        url = f"http://{self._remote_state.host}/models/{self._remote_state.project_name}/{__name}"
        if inspect.isfunction(getattr(ak.AutoModel, __name, None)):

            def perform_method(*args, **kwargs):
                return _perform_request(
                    "POST",
                    url,
                    *args,
                    **kwargs,
                )

            return perform_method
        else:
            return _perform_request(
                "GET",
                url,
            )


def patch_automodel() -> None:
    __AutoModel__init__ = ak.AutoModel.__init__

    def _AutoModel__init__(
        self: AutoModel,
        *args: Any,
        optimizer: Union[str, hyperparameters.Choice, None] = None,
        learning_rate: Union[float, hyperparameters.Choice, None] = None,
        host: Union[str, tuple[str, int], None] = None,
        **kwargs: Any,
    ) -> None:
        if host:
            _host = host if isinstance(host, str) else f"{host[0]}:{host[1]}"
            self._remote_state = AutoModel.RemoteState(
                host=_host,
                project_name=cast(
                    str,
                    _perform_request(
                        "PUT",
                        f"http://{_host}/models",
                        *args,
                        timeout=3,
                        optimizer=optimizer,
                        learning_rate=learning_rate,
                        **kwargs,
                    ),
                ),
            )
        else:
            __AutoModel__init__(self, *args, **kwargs)  # type: ignore
            setattr(
                self.tuner.hypermodel,
                "_compile_choices",
                {
                    name: (
                        choice
                        if isinstance(choice, hyperparameters.Choice)
                        else hyperparameters.Choice(name, [choice])
                    )
                    for name, choice in (
                        ("optimizer", optimizer),
                        ("learning_rate", learning_rate),
                    )
                    if choice is not None
                },
            )

    AutoModel.__init__ = _AutoModel__init__


patch_automodel()


_client = httpx.Client()


def _perform_request(
    method: str,
    url: str,
    *args: object,
    timeout: httpx._types.TimeoutTypes = None,
    **kwargs: object,
) -> object:
    with _client.stream(
        method,
        url,
        content=pickle.dumps((args, kwargs)),
        timeout=timeout,
    ) as response:
        delimiter = b":::::"
        type = b"start"
        content = b""
        response.aiter_raw()

        for chunk in chain(response.iter_raw(), [b"end" + delimiter]):
            if delimiter not in chunk:
                content += chunk
                continue

            if type == b"start":
                pass
            elif type == b"stdout":
                sys.stdout.write(content.decode("utf-8"))
            elif type == b"stderr":
                sys.stderr.write(content.decode("utf-8"))
            elif type == b"result":
                result = pickle.loads(content)
                if isinstance(result, BaseException):
                    raise result
                return pickle.loads(content)
            else:
                raise Exception(f"Unexpected content type: {type.decode('utf-8')}")

            [type, content] = chunk.split(delimiter)

        raise Exception("Never received result")


def patch_graph() -> None:
    __compile_keras_model = Graph._compile_keras_model

    def _compile_keras_model(self, hp, model):
        __Choice = hp.Choice

        def Choice(name, values, **kwargs):
            choice: Optional[hyperparameters.Choice] = getattr(
                self, "_compile_choices", {}
            ).get(name)
            if choice:
                values = choice.values
                kwargs["ordered"] = choice.ordered
                kwargs["default"] = choice.default
            return __Choice(name, values, **kwargs)

        hp.Choice = Choice

        return __compile_keras_model(self, hp, model)

    Graph._compile_keras_model = _compile_keras_model


patch_graph()
