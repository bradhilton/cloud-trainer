from aiohttp import web
from aiohttp.typedefs import Handler
import asyncio
import autokeras as ak
import autokeras_patch
from hooks import Hook
import nest_asyncio
import pickle
import sys
from tblib import pickling_support
from typing import Any, Callable, Coroutine

autokeras_patch.apply()
nest_asyncio.apply()
pickling_support.install()

models: dict[str, ak.AutoModel] = {}


async def start_server() -> None:
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
    port = 9650

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
        response = web.StreamResponse()
        await response.prepare(request)
        current_loop = asyncio.get_event_loop()
        delimiter = b":::::"

        def write(type: bytes):
            def write(__s: str) -> None:
                current_loop.run_until_complete(
                    response.write(type + delimiter + bytes(__s, "utf-8"))
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
    return getattr(model, request.match_info["method"])(*args, **kwargs)


@stream
async def loop(request: web.Request) -> Any:
    while True:
        try:
            await asyncio.sleep(1)
            print("Loop...")
        except BaseException as exception:
            raise exception
