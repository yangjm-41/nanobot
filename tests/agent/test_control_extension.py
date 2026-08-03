"""Regression tests for the opt-in Coco control extension."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.runner_helpers import make_run_spec
from nanobot.agent.control import (
    CONTROL_EXTENSION_API,
    ToolSuspension,
    ToolSuspensionInfo,
)
from nanobot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMResponse, ToolCallRequest


class _StaticTool(Tool):
    def __init__(self, name: str, result: Any = "ok") -> None:
        self._name = name
        self._result = result
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> Any:
        self.calls += 1
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class _SuspensionCaptureHook(AgentHook):
    def __init__(self) -> None:
        super().__init__(reraise=True)
        self.suspensions: list[ToolSuspensionInfo] = []

    async def on_tool_suspended(
        self,
        context: AgentHookContext,
        suspension: ToolSuspensionInfo,
    ) -> None:
        assert context.stop_reason == "tool_suspended"
        self.suspensions.append(deepcopy(suspension))


def _provider(*responses: LLMResponse) -> MagicMock:
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=list(responses))
    provider.generation = SimpleNamespace(
        temperature=0.1,
        max_tokens=512,
        reasoning_effort=None,
    )
    provider.estimate_prompt_tokens.return_value = (100, "test")
    return provider


def _spec(provider: MagicMock, tools: ToolRegistry, **kwargs: Any):
    return make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "run"}],
        tools=tools,
        model="test-model",
        max_iterations=kwargs.pop("max_iterations", 4),
        max_tool_result_chars=10_000,
        **kwargs,
    )


def test_control_contract_round_trip_is_isolated() -> None:
    assert CONTROL_EXTENSION_API == 1
    source = ToolSuspensionInfo(
        suspension_id="s-1",
        tool_call_id="call-1",
        tool_name="approval",
        arguments={"value": 1},
        assistant_message={"role": "assistant", "tool_calls": []},
        metadata={"kind": "confirm"},
    )
    checkpoint = source.to_checkpoint()
    restored = ToolSuspensionInfo.from_checkpoint(checkpoint)
    checkpoint["metadata"]["kind"] = "changed"
    assert restored.metadata == {"kind": "confirm"}
    assert source.metadata == {"kind": "confirm"}


@pytest.mark.asyncio
async def test_critical_control_hook_errors_propagate() -> None:
    class CriticalHook(AgentHook):
        def __init__(self) -> None:
            super().__init__(reraise=True)

        async def before_final_response(self, context: AgentHookContext) -> None:
            raise RuntimeError("control hook failed")

    with pytest.raises(RuntimeError, match="control hook failed"):
        await CompositeHook([CriticalHook()]).before_final_response(
            AgentHookContext(iteration=0, messages=[])
        )


@pytest.mark.asyncio
async def test_required_directive_narrows_one_iteration_and_retries_one_final() -> None:
    tools = ToolRegistry()
    target = _StaticTool("target")
    tools.register(target)
    tools.register(_StaticTool("other"))
    provider = _provider(
        LLMResponse(content="violating answer"),
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="call-target", name="target", arguments={})],
            finish_reason="tool_calls",
        ),
        LLMResponse(content="accepted answer"),
    )

    class RequiredHook(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            if context.iteration == 0:
                context.directive.required_tool_name_once = "target"

        async def before_final_response(self, context: AgentHookContext) -> None:
            if context.iteration == 0:
                context.directive.required_tool_name_once = "target"
                context.directive.discard_final_response_and_continue = True

    result = await AgentRunner().run(_spec(provider, tools, hook=RequiredHook()))

    assert result.final_content == "accepted answer"
    assert "violating answer" not in str(result.messages)
    assert target.calls == 1
    first, second, third = provider.chat_with_retry.await_args_list
    assert first.kwargs["tool_choice"] == "required"
    assert second.kwargs["tool_choice"] == "required"
    assert [item["function"]["name"] for item in first.kwargs["tools"]] == ["target"]
    assert [item["function"]["name"] for item in second.kwargs["tools"]] == ["target"]
    assert "tool_choice" not in third.kwargs
    assert {item["function"]["name"] for item in third.kwargs["tools"]} == {
        "other",
        "target",
    }


@pytest.mark.asyncio
async def test_unknown_required_tool_fails_before_provider_call() -> None:
    tools = ToolRegistry()
    tools.register(_StaticTool("known"))
    provider = _provider(LLMResponse(content="must not run"))

    class MissingHook(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            context.directive.required_tool_name_once = "missing"

    with pytest.raises(ValueError, match="required tool is unavailable"):
        await AgentRunner().run(_spec(provider, tools, hook=MissingHook()))
    provider.chat_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_suspension_gate_disabled_preserves_normal_tool_error_behavior() -> None:
    tools = ToolRegistry()
    tools.register(_StaticTool("approval", ToolSuspension("s-disabled")))
    provider = _provider(
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="call-1", name="approval", arguments={})],
            finish_reason="tool_calls",
        ),
        LLMResponse(content="recovered"),
    )

    result = await AgentRunner().run(_spec(provider, tools))

    assert result.stop_reason == "completed"
    assert result.suspension is None
    assert result.final_content == "recovered"
    tool_message = next(row for row in result.messages if row.get("role") == "tool")
    assert "ToolSuspension" in tool_message["content"]


@pytest.mark.asyncio
async def test_suspension_gate_checkpoints_before_hook_and_stops_later_tools() -> None:
    tools = ToolRegistry()
    tools.register(_StaticTool("first", "first-result"))
    tools.register(
        _StaticTool(
            "approval",
            ToolSuspension("s-enabled", {"kind": "confirmation"}),
        )
    )
    later = _StaticTool("later", "must-not-run")
    tools.register(later)
    provider = _provider(
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(id="call-first", name="first", arguments={}),
                ToolCallRequest(id="call-approval", name="approval", arguments={}),
                ToolCallRequest(id="call-later", name="later", arguments={}),
            ],
            finish_reason="tool_calls",
        )
    )
    checkpoints: list[dict[str, Any]] = []
    hook = _SuspensionCaptureHook()

    async def checkpoint(payload: dict[str, Any]) -> None:
        checkpoints.append(deepcopy(payload))

    result = await AgentRunner().run(
        _spec(
            provider,
            tools,
            hook=hook,
            allow_tool_suspension=True,
            concurrent_tools=True,
            checkpoint_callback=checkpoint,
        )
    )

    assert result.stop_reason == "tool_suspended"
    assert result.final_content is None
    assert result.suspension is not None
    assert result.suspension.suspension_id == "s-enabled"
    assert later.calls == 0
    assert checkpoints[-1]["phase"] == "tool_suspended"
    assert checkpoints[-1]["pending_tool_calls"][0]["id"] == "call-approval"
    assert checkpoints[-1]["completed_tool_results"][0]["tool_call_id"] == "call-first"
    assert checkpoints[-1]["skipped_tool_results"][0]["tool_call_id"] == "call-later"
    assert hook.suspensions[0].suspension_id == "s-enabled"


@pytest.mark.asyncio
async def test_loop_suspends_and_continues_without_synthetic_user(loop_factory) -> None:
    tools = ToolRegistry()
    tools.register(_StaticTool("approval", ToolSuspension("s-loop")))
    provider = _provider(
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call-loop",
                    name="approval",
                    arguments={"question": "continue?"},
                )
            ],
            finish_reason="tool_calls",
        ),
        LLMResponse(content="continued"),
    )
    provider.get_default_model.return_value = "test-model"
    loop = loop_factory(provider=provider)
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=None)
    hook = _SuspensionCaptureHook()

    suspended = await loop.process_direct(
        "start",
        session_key="api:control",
        channel="api",
        chat_id="control",
        hooks=[hook],
        tools=tools,
        allow_tool_suspension=True,
    )

    assert suspended is None
    session = loop.sessions.get_or_create("api:control")
    assert session.metadata[loop._RUNTIME_CHECKPOINT_KEY]["phase"] == "tool_suspended"
    assert session.messages[-1]["role"] == "assistant"
    assert session.messages[-1]["tool_calls"][0]["id"] == "call-loop"

    resumed = await loop.continue_tool_result(
        session_key="api:control",
        suspension_id="s-loop",
        tool_result={"approved": True},
        channel="api",
        chat_id="control",
        hooks=[hook],
        tools=tools,
    )

    assert resumed is not None
    assert resumed.content == "continued"
    assert loop._RUNTIME_CHECKPOINT_KEY not in session.metadata
    second_request = provider.chat_with_retry.await_args_list[1].kwargs["messages"]
    assistant_index = next(
        index
        for index, row in enumerate(second_request)
        if row.get("role") == "assistant" and row.get("tool_calls")
    )
    assert second_request[assistant_index + 1] == {
        "role": "tool",
        "content": '{"approved":true}',
        "tool_call_id": "call-loop",
        "name": "approval",
    }
    assert not any(row.get("role") == "user" for row in second_request[assistant_index + 1 :])


@pytest.mark.asyncio
async def test_loop_rejects_conflicting_suspension_id(loop_factory) -> None:
    tools = ToolRegistry()
    tools.register(_StaticTool("approval", ToolSuspension("s-real")))
    provider = _provider(
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="call-real", name="approval", arguments={})],
            finish_reason="tool_calls",
        )
    )
    provider.get_default_model.return_value = "test-model"
    loop = loop_factory(provider=provider)
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=None)
    await loop.process_direct(
        "start",
        session_key="api:conflict",
        tools=tools,
        allow_tool_suspension=True,
    )

    with pytest.raises(ValueError, match="does not match"):
        await loop.continue_tool_result(
            session_key="api:conflict",
            suspension_id="s-wrong",
            tool_result="no",
            tools=tools,
        )
