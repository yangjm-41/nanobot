from unittest.mock import MagicMock

from agent.runner_helpers import make_run_spec
from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMResponse, ToolCallRequest

ROUTE_TOOL_NAME = "route_capabilities"


class _RouteTool(Tool):
    @property
    def name(self) -> str:
        return ROUTE_TOOL_NAME

    @property
    def description(self) -> str:
        return "route"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **_kwargs):
        return "no_match"


class _RouteBeforeFinalHook(AgentHook):
    def __init__(self) -> None:
        super().__init__()
        self._checked = False

    async def before_final_response(self, context: AgentHookContext) -> None:
        if self._checked:
            return
        self._checked = True
        context.directive.required_tool_name_once = ROUTE_TOOL_NAME
        context.directive.discard_final_response_and_continue = True


async def test_iteration_directive_discards_candidate_and_forces_one_tool_call() -> None:
    provider = MagicMock()
    requests: list[dict] = []

    async def chat_with_retry(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            return LLMResponse(content="candidate reply")
        if len(requests) == 2:
            return LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCallRequest("route-call", ROUTE_TOOL_NAME, {})],
            )
        return LLMResponse(content="final reply")

    provider.chat_with_retry = chat_with_retry
    tools = ToolRegistry()
    tools.register(_RouteTool())
    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "complex request"}],
        tools=tools,
        model="test-model",
        max_iterations=3,
        max_tool_result_chars=AgentDefaults().max_tool_result_chars,
        hook=_RouteBeforeFinalHook(),
    ))

    assert result.final_content == "final reply"
    assert all(
        message.get("content") != "candidate reply"
        for message in result.messages
    )
    assert requests[1]["tool_choice"] == {
        "type": "function",
        "function": {"name": ROUTE_TOOL_NAME},
    }
    assert "tool_choice" not in requests[2]
