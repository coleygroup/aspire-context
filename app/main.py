from fastapi import FastAPI

from app.v1.api.endpoints import router as v1_router
from app.v2.api.endpoints import router as v2_router


app = FastAPI()

app.include_router(v1_router, prefix="/api/v1")
app.include_router(v2_router, prefix="/api/v2")
