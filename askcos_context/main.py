from fastapi import FastAPI

from askcos_context.v1.api.endpoints import router as v1_router
from askcos_context.v2.api.endpoints import router as v2_router


app = FastAPI()

app.include_router(v1_router, prefix="/api/v1")
app.include_router(v2_router, prefix="/api/v2")
