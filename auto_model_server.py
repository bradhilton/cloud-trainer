from aiohttp import web
from aiohttp.typedefs import Handler
import asyncio
import autokeras as ak
import autokeras_patch
from hooks import Hook
import httpx
import nest_asyncio
import os
import pickle
import re
import sys
from tblib import pickling_support
from typing import Any, Callable, Coroutine, Literal

autokeras_patch.apply()
nest_asyncio.apply()
pickling_support.install()

models: dict[str, ak.AutoModel] = {}
_init_args_and_kwargs: dict[str, tuple[list[Any], dict[str, Any]]] = {}
_tuner0_port = 9101
_child_tasks: list[asyncio.Task] = []

TUNER_ID = Literal["chief", "tuner0", "tuner1", "tuner2"]


async def start_server(*, tuner_id: TUNER_ID) -> None:
    match = re.match(r"chief|tuner(\d+)", tuner_id)
    assert match, 'tuner_id must be "chief" or "tuner0" or "tuner1" or "tuner2"'

    os.environ["KERASTUNER_TUNER_ID"] = tuner_id
    os.environ["KERASTUNER_HOST"] = "localhost"
    os.environ["KERASTUNER_PORT"] = str(_tuner0_port - 2)
    os.environ["CUDA_VISIBLE_DEVICES"] = match.group(1) or "3"

    app = web.Application(client_max_size=1024**3)
    app.add_routes(
        [
            web.get("/models/{project_name}/{attribute}", get_model_attribute),
            web.post("/models/{project_name}/{method}", perform_model_method),
            web.put("/models", get_or_create_model),
            web.post("/loop", loop),
        ]
    )
    runner = web.AppRunner(app)
    await runner.setup()

    host = "0.0.0.0"
    port = _tuner0_port + int(match.group(1) or "-1")

    site = web.TCPSite(
        runner,
        host=host,
        port=port,
    )
    await site.start()

    print("AutoModel server started at", f"http://{host}:{port}")

    while True:
        try:
            await asyncio.sleep(0.25)
        except BaseException:
            break

    await site.stop()


def stream(handler: Callable[[web.Request], Coroutine[Any, Any, Any]]) -> Handler:
    async def stream(request: web.Request) -> web.StreamResponse:
        quiet = "quiet" in request.query
        response = web.StreamResponse()
        await response.prepare(request)
        current_loop = asyncio.get_event_loop()
        delimiter = b":::::"

        def write(type: bytes):
            def write(__s: str) -> None:
                current_loop.run_until_complete(
                    response.write(
                        b"heartbeat" + delimiter
                        if quiet
                        else type + delimiter + bytes(__s, "utf-8")
                    )
                )

            return write

        with Hook(sys.stdout.write, write(b"stdout")):
            with Hook(sys.stderr.write, write(b"stderr")):
                try:
                    result = await handler(request)
                except BaseException as exception:
                    result = exception

        if isinstance(result, (asyncio.CancelledError, ConnectionResetError)):
            print("Request Cancelled ❌", file=sys.stderr)
            global _child_tasks
            for task in _child_tasks:
                task.cancel()
            return response

        await response.write(b"result" + delimiter + pickle.dumps(result))
        return response

    return stream


@stream
async def get_or_create_model(request: web.Request) -> str:
    body = await request.read()
    args, kwargs = pickle.loads(body)
    model = ak.AutoModel(*args, **kwargs)
    models[model.project_name] = model
    _init_args_and_kwargs[model.project_name] = (args, kwargs)
    return model.project_name


@stream
async def get_model_attribute(request: web.Request) -> Any:
    return getattr(
        models[request.match_info["project_name"]], request.match_info["attribute"]
    )


@stream
async def perform_model_method(request: web.Request) -> Any:
    project_name = request.match_info["project_name"]
    model = models[project_name]
    args, kwargs = pickle.loads(await request.read())

    if (
        request.match_info["method"] == "fit"
        and os.environ["KERASTUNER_TUNER_ID"] == "chief"
    ):
        global _child_tasks
        _child_tasks = [
            asyncio.create_task(fit_tuner(project_name, tuner_id, (args, kwargs)))
            for tuner_id in range(3)
        ]
    return getattr(model, request.match_info["method"])(*args, **kwargs)


_client = httpx.AsyncClient()


async def fit_tuner(
    project_name: str,
    tuner_id: int,
    fit_args_and_kwargs: tuple[list[Any], dict[str, Any]],
) -> None:
    await _client.put(
        f"http://0.0.0.0:{_tuner0_port + tuner_id}/models?quiet",
        content=pickle.dumps(_init_args_and_kwargs[project_name]),
    )
    await _client.post(
        f"http://0.0.0.0:{_tuner0_port + tuner_id}/models/{project_name}/fit?quiet",
        content=pickle.dumps(fit_args_and_kwargs),
        timeout=None,
    )


@stream
async def loop(request: web.Request) -> Any:
    while True:
        try:
            await asyncio.sleep(1)
            print("Loop...")
        except BaseException as exception:
            raise exception
