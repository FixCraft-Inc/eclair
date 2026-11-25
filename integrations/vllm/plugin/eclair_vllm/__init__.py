# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Opt-in vLLM general plugin for running compressed-tensors W4A16 checkpoints on sm_70.

vLLM loads `register` in every engine process through the `vllm.general_plugins`
entry point. It registers `eclair-ct-sm70` and the experimental
`eclair-gptq-sm70`, which a launch selects explicitly with `--quantization`, and the CUSTOM
attention backend, selected with `--attention-backend CUSTOM`. Nothing changes
for launches that name neither, except two vLLM-sanctioned out-of-tree custom
ops: a build without vLLM's FlashAttention extension gets a working
vision-encoder rotary op (`vision_sm70`), and Gated DeltaNet prefill uses
Eclair's native state pass when libeclair_kernels is installed (`gdn_sm70`). Its
native output pass also needs ECLAIR_NATIVE_GDN_OUTPUT=1; FLA output remains the
default pending coding qualification. Missing native kernels use vLLM's own path.
The plugin is pinned to one vLLM release
because it subclasses that release's compressed-tensors classes. Another
release leaves the method unregistered, so a launch that names it fails.
"""

from __future__ import annotations

import logging
import os

SUPPORTED_VLLM = "0.32.0"
METHOD = "eclair-ct-sm70"
CONTRACT = "eclair.vllm-ct-sm70/1"

logger = logging.getLogger(__name__)


def register() -> None:
    import vllm

    version = vllm.__version__.split("+", 1)[0]
    if version != SUPPORTED_VLLM:
        logger.warning("Eclair %s is pinned to vLLM %s; found %s, not registering",
                       METHOD, SUPPORTED_VLLM, vllm.__version__)
        return
    from . import attention_sm70, gdn_sm70, gptq_sm70, vision_sm70, wna16_sm70

    if os.environ.get("ECLAIR_MTP_DRAFT_VOCAB"):
        from . import mtp_sm70

        mtp_sm70.register()
    wna16_sm70.register()
    gptq_sm70.register()
    # Selected only by `--attention-backend CUSTOM`. Other launches keep vLLM's choice.
    attention_sm70.register()
    vision_sm70.register()
    gdn_sm70.register()
