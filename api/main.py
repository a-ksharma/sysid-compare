"""FastAPI application entry point for the system identification service."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.deps import load_application_state
from api.routers import analysis, health, models, simulate, systems

# Logging configuration
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan handler: loads physical plants and models once at startup."""
    logger.info("Starting up FastAPI application...")
    load_application_state()
    yield
    logger.info("Shutting down FastAPI application...")


app = FastAPI(
    title="System Identification Comparison API",
    version="0.1.0",
    description=(
        "Production REST API comparing classical (ARX, Prony) and learned "
        "(Neural ODE) system identification models across RC low-pass, DC "
        "motor, and mass-spring-damper plants."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS configuration
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Standardized error handlers conforming to Spec Section 10:
# {"error": {"code": "<machine_code>", "message": "<human text>"}}
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Format HTTPExceptions into standard error schema."""
    if (
        isinstance(exc.detail, dict)
        and "code" in exc.detail
        and "message" in exc.detail
    ):
        error_body = exc.detail
    else:
        error_body = {"code": "HTTP_ERROR", "message": str(exc.detail)}
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error_body},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Format Pydantic RequestValidationErrors into standard error schema."""
    first_error = exc.errors()[0] if exc.errors() else {}
    msg = first_error.get("msg", "Validation error.")
    loc = " -> ".join(str(loc_item) for loc_item in first_error.get("loc", []))
    formatted_message = f"Invalid input at '{loc}': {msg}" if loc else msg

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": formatted_message,
            }
        },
    )


# Include API v1 routers
API_V1_PREFIX = "/api/v1"
app.include_router(health.router, prefix=API_V1_PREFIX)
app.include_router(systems.router, prefix=API_V1_PREFIX)
app.include_router(models.router, prefix=API_V1_PREFIX)
app.include_router(simulate.router, prefix=API_V1_PREFIX)
app.include_router(analysis.router, prefix=API_V1_PREFIX)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Root redirect message pointing to /docs and health check."""
    return {
        "message": (
            "System Identification API is running. "
            "Explore OpenAPI documentation at /docs"
        ),
        "health": "/api/v1/health",
    }
