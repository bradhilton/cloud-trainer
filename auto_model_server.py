from aiohttp import web
from aiohttp.typedefs import Handler
import asyncio
import autokeras as ak
import autokeras_patch
from hooks import Hook
import httpx
from multiprocessing import Process
import nest_asyncio
import os
import pickle
import sys
from tblib import pickling_support
from typing import Any, Callable, Coroutine

autokeras_patch.apply()
nest_asyncio.apply()
pickling_support.install()

models: dict[str, ak.AutoModel] = {}
_init_args_and_kwargs: dict[str, tuple[list[Any], dict[str, Any]]] = {}
_tuner0_port = 9651
_child_tasks: list[asyncio.Task] = []


async def start_server(*, tuner_id: int) -> None:
    os.environ["KERASTUNER_TUNER_ID"] = f"tuner{tuner_id}"
    os.environ["KERASTUNER_ORACLE_IP"] = "localhost"
    os.environ["KERASTUNER_ORACLE_PORT"] = str(_tuner0_port - 1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(tuner_id)

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
    port = _tuner0_port + tuner_id

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
        and os.environ["KERASTUNER_TUNER_ID"] == "tuner0"
    ):
        start_chief_process(project_name)

        global _child_tasks
        _child_tasks = [
            await fit_tuner(project_name, tuner_id, (args, kwargs))
            for tuner_id in range(1, 4)
        ]

    return getattr(model, request.match_info["method"])(*args, **kwargs)


_chief_process = Process()


def chief_process_target(*args: Any, **kwargs: Any) -> None:
    print("Started chief process")
    os.environ["KERASTUNER_TUNER_ID"] = "chief"
    ak.AutoModel(*args, **kwargs)


def start_chief_process(project_name: str) -> None:
    global _chief_process
    if _chief_process.is_alive():
        _chief_process.kill()

    args, kwargs = _init_args_and_kwargs[project_name]

    _chief_process = Process(target=chief_process_target, args=args, kwargs=kwargs)
    _chief_process.start()


_client = httpx.AsyncClient()


async def noop() -> None:
    pass


async def fit_tuner(
    project_name: str,
    tuner_id: int,
    fit_args_and_kwargs: tuple[list[Any], dict[str, Any]],
) -> asyncio.Task:
    try:
        await _client.put(
            f"http://0.0.0.0:{_tuner0_port + tuner_id}/models?quiet",
            content=pickle.dumps(_init_args_and_kwargs[project_name]),
        )
        response = await _client.send(
            _client.build_request(
                "POST",
                f"http://0.0.0.0:{_tuner0_port + tuner_id}/models/{project_name}/fit?quiet",
                content=pickle.dumps(fit_args_and_kwargs),
                timeout=None,
            ),
            stream=True,
        )
        return asyncio.create_task(response.aread())
    except:
        print(f"Unable to reach tuner{tuner_id}", file=sys.stderr)
        return asyncio.create_task(noop())


@stream
async def loop(request: web.Request) -> Any:
    while True:
        try:
            await asyncio.sleep(1)
            print("Loop...")
        except BaseException as exception:
            raise exception
