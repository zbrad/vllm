# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tuned-builds extension for vllm/platforms/interface.py: the warning for
environment variables vLLM no longer reads. Kept out of interface.py so upstream
merges of that file stay conflict-free; `Platform.validate_environ` only imports
`InterfaceExt` and skips any variable it reports as handled.
"""

from vllm.ext.envs_ext import EnvsExt
from vllm.logger import init_logger

logger = init_logger(__name__)


class InterfaceExt:
    """Removed-environment-variable handling for Platform.validate_environ."""

    @classmethod
    def warn_if_removed_env(cls, env: str) -> bool:
        """Warn when ``env`` is a variable vLLM no longer reads.

        A removed variable is a no-op, not a typo, so the caller must not treat
        it like an unrecognized ``VLLM_`` variable, even with ``hard_fail``.

        Args:
            env: Name of an environment variable.

        Returns:
            True if ``env`` is a removed variable and a warning was logged.
        """
        replacement = EnvsExt.REMOVED_ENVIRONMENT_VARIABLES.get(env)
        if replacement is None:
            return False
        logger.warning(
            "Environment variable %s is no longer read by vLLM and "
            "has no effect; use %s instead.",
            env,
            replacement,
        )
        return True
