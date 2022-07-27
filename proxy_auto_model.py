import autokeras as ak
import httpx
import inspect
from IPython.display import clear_output
from itertools import chain
import pickle
import sys
from typing import Any


class ProxyAutoModel:
    __url__: str

    def __init__(
        self, *, host: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        project_name = _perform_request(
            "PUT",
            f"http://{host}/models",
            *args,
            timeout=3,
            **kwargs,
        )
        self.__url__ = f"http://{host}/models/{project_name}"

    def __getattribute__(self, __name: str) -> Any:
        if __name.startswith("__"):
            return super().__getattribute__(__name)

        url = f"{self.__url__}/{__name}"

        if inspect.isfunction(getattr(ak.AutoModel, __name, None)):
            return lambda *args, **kwargs: _perform_request(
                "POST", url, *args, **kwargs
            )
        else:
            return _perform_request(
                "GET",
                url,
            )


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
                decoded_content = content.decode("utf-8")
                if decoded_content.startswith("\033[2J"):
                    clear_output(wait=True)
                sys.stdout.write(decoded_content)
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
