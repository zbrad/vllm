# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tuned-builds extension for vllm/envs.py: environment variables vLLM no longer
reads. Kept out of envs.py so upstream merges of that file stay conflict-free;
the only reader is `InterfaceExt` in vllm/ext/platforms/interface_ext.py.
"""


class EnvsExt:
    """Fork additions to vllm/envs.py."""

    # Env vars vLLM used to read that were removed in favor of a CLI flag or a
    # different env var. Setting one now has no effect; the generic "unknown
    # variable" warning in Platform.validate_environ
    # (vllm/platforms/interface.py) doesn't say why, so call these out by name.
    REMOVED_ENVIRONMENT_VARIABLES: dict[str, str] = {
        "VLLM_ATTENTION_BACKEND": "the --attention-backend CLI flag",
    }
