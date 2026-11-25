# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""CPU HTTP contract tests, run in the pinned vLLM environment; no CUDA requests."""

import argparse
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

try:
    import httpx
    from fastapi import FastAPI
    from starlette.datastructures import State
    from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware
except ImportError as error:
    raise unittest.SkipTest("requires the pinned vLLM Python environment") from error

sys.path.insert(0, str(Path(__file__).parent / "control_plugin"))
from eclair_vllm_control import CacheControl


class CacheControlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.plugin = CacheControl()
        self.app = FastAPI()
        self.plugin.attach_router(self.app)
        self.app.add_middleware(AuthenticationMiddleware, tokens=["private-backend-key"])
        self.engine = AsyncMock()
        self.engine.reset_prefix_cache.return_value = True
        self.args = argparse.Namespace(host="127.0.0.1", api_key=["private-backend-key"])
        await self.plugin.init_state(self.engine, self.app.state, self.args)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                        base_url="http://127.0.0.1", headers={
                                            "Authorization": "Bearer private-backend-key"})

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_authentication_guards_both_routes(self) -> None:
        for method, path in (("GET", "/v1/eclair/cache"), ("POST", "/v1/eclair/cache/reset")):
            response = await self.client.request(method, path, headers={"Authorization": "Bearer wrong"})
            self.assertEqual(response.status_code, 401)
        self.engine.reset_prefix_cache.assert_not_called()

    async def test_stream_framing_does_not_allow_an_unbounded_body(self) -> None:
        response = await self.client.post("/v1/eclair/cache/reset", content=b"{}",
                                          headers={"Transfer-Encoding": "chunked"})
        self.assertEqual(response.status_code, 400)
        response = await self.client.post("/v1/eclair/cache/reset", content=b" " * 10000,
                                          headers={"Content-Length": "2"})
        self.assertEqual(response.status_code, 400)
        self.engine.reset_prefix_cache.assert_not_called()

    async def test_only_explicit_idle_prefix_reset_is_exposed(self) -> None:
        response = await self.client.get("/v1/eclair/cache")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["release_vram"])
        response = await self.client.post("/v1/eclair/cache/reset", content=b"{}")
        self.assertEqual((response.status_code, response.json()), (200, {"success": True}))
        self.engine.reset_prefix_cache.assert_awaited_once_with(
            reset_running_requests=False, reset_connector=False)
        for path in ("/reset_prefix_cache", "/collective_rpc", "/sleep", "/v1/eclair/unload"):
            self.assertEqual((await self.client.post(path)).status_code, 404)

    async def test_forced_external_and_oversized_requests_never_reach_the_engine(self) -> None:
        for path, content in (("/v1/eclair/cache/reset?reset_running_requests=true", b"{}"),
                              ("/v1/eclair/cache/reset?reset_external=true", b"{}"),
                              ("/v1/eclair/cache/reset", b'{"force":true}'),
                              ("/v1/eclair/cache/reset", b" " * 10000)):
            response = await self.client.post(path, content=content)
            self.assertEqual(response.status_code, 400)
        self.engine.reset_prefix_cache.assert_not_called()

    async def test_engine_busy_and_invalid_success_are_preserved(self) -> None:
        self.engine.reset_prefix_cache.return_value = False
        response = await self.client.post("/v1/eclair/cache/reset", content=b"{}")
        self.assertEqual((response.status_code, response.json()), (409, {"success": False}))
        self.engine.reset_prefix_cache.return_value = 1
        self.assertEqual((await self.client.post("/v1/eclair/cache/reset", content=b"{}")).status_code, 502)

    async def test_init_refuses_missing_authentication_network_bind_or_wrong_version(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            for keys in (None, [], [""]):
                with self.subTest(keys=keys), self.assertRaises(RuntimeError):
                    await self.plugin.init_state(self.engine, State(), argparse.Namespace(
                        host="127.0.0.1", api_key=keys))
        with self.assertRaises(RuntimeError):
            await self.plugin.init_state(self.engine, State(), argparse.Namespace(
                host="0.0.0.0", api_key=["key"]))
        with patch("eclair_vllm_control.importlib.metadata.version", return_value="0.33.0"):
            with self.assertRaises(RuntimeError):
                await self.plugin.init_state(self.engine, State(), self.args)


if __name__ == "__main__":
    unittest.main()
