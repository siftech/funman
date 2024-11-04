"""
Definitions for the REST API endpoints.  Running this file will start a uvicorn server that serves the API.

Raises
------
HTTPException
    HTTPException description
"""

import logging
import traceback
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Optional, Union

import uvicorn

l = logging.getLogger(__name__)
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Security,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyHeader
from typing_extensions import Annotated

from funman import __version__ as FunmanVersion
from funman.api.settings import Settings
from funman.model.bilayer import BilayerModel
from funman.model.generated_models.petrinet import Model as GeneratedPetriNet
from funman.model.generated_models.regnet import Model as GeneratedRegNet
from funman.model.model import _wrap_with_internal_model
from funman.model.petrinet import PetrinetModel
from funman.model.regnet import RegnetModel
from funman.server.exception import NotFoundFunmanException
from funman.server.query import (
    FunmanResults,
    FunmanWorkRequest,
    FunmanWorkUnit,
)
from funman.server.storage import Storage
from funman.server.worker import FunmanWorker

settings = Settings()
_storage = Storage()
_worker = FunmanWorker(_storage)


# Rig some services to run while the API is online
@asynccontextmanager
async def lifespan(_: FastAPI):
    _storage.start(settings.data_path)
    _worker.start()
    yield
    _worker.stop()
    _storage.stop()


def get_storage():
    return _storage


def get_worker():
    return _worker


def _key_auth(api_key: str, token: str, *, name: str = "API"):
    # bypass key auth if no token is provided
    if api_key is None:
        l.warning(f"WARNING: Running without {name} token")
        return

    # ensure the token is a non-empty string
    if not isinstance(api_key, str) or api_key == "":
        print(f"ERROR: {name} token is either empty or not a string")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )

    if token != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Forbidden"
        )


api_key_header = APIKeyHeader(name="token", auto_error=False)
admin_key_header = APIKeyHeader(name="token", auto_error=False)


def _api_key_auth(token: str = Security(api_key_header)):
    return _key_auth(settings.funman_api_token, token)


def _admin_key_auth(token: str = Security(admin_key_header)):
    return _key_auth(settings.funman_admin_token, token, name="Admin API")


if settings.funman_base_url is not None:
    print(f"FUNMAN_BASE_URL={settings.funman_base_url}")

app = FastAPI(
    title="funman_api", lifespan=lifespan, root_path=settings.funman_base_url
)
# router for public api
api_router = APIRouter(
    prefix="/api",
    dependencies=[Depends(_api_key_auth)],
)
# router for admin api
# TODO this is a placeholder
admin_router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(_admin_key_auth)],
    include_in_schema=False,
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="FUNMAN API",
        version=FunmanVersion,
        description="Functional Model Analysis",
        routes=app.routes,
        servers=app.servers,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@contextmanager
def internal_error_handler():
    eid = uuid.uuid4()
    try:
        yield
    except HTTPException:
        raise
    except Exception:
        l.error(f"Internal Server Error ({eid}):")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {eid}"
        )


@app.get("/", include_in_schema=False)
def read_root(request: Request):
    return RedirectResponse(url=f"{request.scope.get('root_path')}/docs")


@api_router.get(
    "/queries/{query_id}/halt",
    response_model=str,
)
async def halt(
    query_id: str, worker: Annotated[FunmanWorker, Depends(get_worker)]
):
    with internal_error_handler():
        worker.halt(query_id)
        return "Success"


@admin_router.get(
    "/queries/current",
    response_model=Optional[str],
)
async def get_current(worker: Annotated[FunmanWorker, Depends(get_worker)]):
    with internal_error_handler():
        return worker.get_current()


@api_router.get(
    "/queries/{query_id}",
    response_model=FunmanResults,
    response_model_exclude_defaults=True,
)
async def get_queries(
    query_id: str, worker: Annotated[FunmanWorker, Depends(get_worker)]
):
    with internal_error_handler():
        try:
            return worker.get_results(query_id)
        except NotFoundFunmanException as e:
            raise HTTPException(404, detail=str(e))


@api_router.post(
    "/queries",
    response_model=FunmanWorkUnit,
    response_model_exclude_defaults=True,
)
async def post_queries(
    model: Union[
        GeneratedPetriNet,
        GeneratedRegNet,
        RegnetModel,
        PetrinetModel,
        # DecapodeModel,
        BilayerModel,
    ],
    request: FunmanWorkRequest,
    worker: Annotated[FunmanWorker, Depends(get_worker)],
):
    with internal_error_handler():
        return worker.enqueue_work(_wrap_with_internal_model(model), request)


# include routers
app.include_router(api_router)
app.include_router(admin_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8190)
