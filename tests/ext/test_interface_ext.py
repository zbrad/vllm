# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform.validate_environ's handling of removed env vars (InterfaceExt)."""

import os
from unittest.mock import patch

import pytest

from vllm.platforms.interface import Platform


class TestValidateEnviron:
    """Test cases for Platform.validate_environ's handling of removed env vars."""

    def test_removed_var_warns_with_actionable_message(self, caplog_vllm):
        # vllm's own logger has propagate=False (see vllm/logger.py), so the
        # plain `caplog` fixture -- which listens on the root logger -- never
        # sees these records. caplog_vllm (tests/conftest.py) temporarily
        # re-enables propagation for the duration of the test.
        with (
            patch.dict(os.environ, {"VLLM_ATTENTION_BACKEND": "flashinfer"}),
            caplog_vllm.at_level("WARNING", logger="vllm"),
        ):
            Platform.validate_environ(hard_fail=False)
        assert any(
            "VLLM_ATTENTION_BACKEND is no longer read by vLLM" in message
            and "--attention-backend" in message
            for message in caplog_vllm.messages
        )

    def test_removed_var_does_not_raise_even_with_hard_fail(self, caplog_vllm):
        """Removed vars are a no-op, not a typo -- hard_fail must not treat
        them the same as an unrecognized VLLM_ variable."""
        with (
            patch.dict(os.environ, {"VLLM_ATTENTION_BACKEND": "flashinfer"}),
            caplog_vllm.at_level("WARNING", logger="vllm"),
        ):
            Platform.validate_environ(hard_fail=True)
        assert any(
            "VLLM_ATTENTION_BACKEND is no longer read by vLLM" in message
            for message in caplog_vllm.messages
        )

    def test_unrecognized_var_still_raises_with_hard_fail(self):
        with (
            patch.dict(os.environ, {"VLLM_THIS_IS_NOT_A_REAL_VAR": "1"}),
            pytest.raises(ValueError, match="Unknown vLLM environment variable"),
        ):
            Platform.validate_environ(hard_fail=True)

    def test_unrecognized_var_warns_without_hard_fail(self, caplog_vllm):
        with (
            patch.dict(os.environ, {"VLLM_THIS_IS_NOT_A_REAL_VAR": "1"}),
            caplog_vllm.at_level("WARNING", logger="vllm"),
        ):
            Platform.validate_environ(hard_fail=False)
        assert any(
            "Unknown vLLM environment variable detected: VLLM_THIS_IS_NOT_A_REAL_VAR"
            in message
            for message in caplog_vllm.messages
        )
