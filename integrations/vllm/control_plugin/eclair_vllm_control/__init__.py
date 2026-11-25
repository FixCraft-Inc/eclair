# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Only the supported EngineClient cache-reset contract; no development API or RPC."""

from __future__ import annotations

import asyncio
import importlib.metadata
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class CacheControl:
    name = "eclair_cache_control"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        @app.get("/v1/eclair/cache")
        async def capability() -> dict:
            return {"schema": "eclair.vllm-cache-control/1", "prefix_reset": True,
                    "reset_running_requests": False, "release_vram": False}

        @app.post("/v1/eclair/cache/reset")
        async def reset(request: Request) -> JSONResponse:
            # A closed, bounded body prevents accidental forwarding of force/external knobs.
            if (request.query_params or request.headers.getlist("content-length") != ["2"]
                    or request.headers.get("transfer-encoding") is not None):
                return JSONResponse({"error": "invalid_reset_request"}, status_code=400)
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > 2:
                    return JSONResponse({"error": "invalid_reset_request"}, status_code=400)
                body.extend(chunk)
            if body != b"{}":
                return JSONResponse({"error": "invalid_reset_request"}, status_code=400)
            engine = getattr(request.app.state, "eclair_cache_engine", None)
            if engine is None:
                return JSONResponse({"error": "engine_unavailable"}, status_code=503)
            try:
                success = await asyncio.wait_for(engine.reset_prefix_cache(
                    reset_running_requests=False, reset_connector=False), timeout=10)
            except TimeoutError:
                return JSONResponse({"error": "reset_timeout"}, status_code=503)
            if type(success) is not bool:
                return JSONResponse({"error": "invalid_reset_result"}, status_code=502)
            return JSONResponse({"success": success}, status_code=200 if success else 409)

    async def init_state(self, engine_client, state, args) -> None:
        if importlib.metadata.version("vllm").split("+", 1)[0] != "0.32.0":
            raise RuntimeError("cache control is pinned to vLLM 0.32.0")
        # vLLM's authentication middleware guards /v1 routes. Refuse to start without it.
        keys = getattr(args, "api_key", None) or [os.environ.get("VLLM_API_KEY")]
        if (engine_client is None or args.host != "127.0.0.1"
                or not any(type(key) is str and key for key in keys)):
            raise RuntimeError("cache control requires a loopback engine with API authentication")
        state.eclair_cache_engine = engine_client
