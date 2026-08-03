"""Opt-in control primitives for embedding the agent runner.

The core agent does not attach product semantics to these values.  Embedders
may use them to constrain one model iteration or to suspend a tool call for an
out-of-process continuation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

CONTROL_EXTENSION_API = 1


@dataclass(slots=True)
class AgentIterationDirective:
    """Mutable instructions that apply to one runner iteration."""

    required_tool_name_once: str | None = None
    discard_final_response_and_continue: bool = False


@dataclass(slots=True)
class ToolSuspensionInfo:
    """Serializable description of one suspended model tool call."""

    suspension_id: str
    tool_call_id: str
    tool_name: str
    arguments: Any
    assistant_message: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_checkpoint(self) -> dict[str, Any]:
        """Return an isolated payload suitable for session persistence."""
        return {
            "suspension_id": self.suspension_id,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": deepcopy(self.arguments),
            "assistant_message": deepcopy(self.assistant_message),
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> ToolSuspensionInfo:
        """Validate and rebuild suspension information from a checkpoint."""
        suspension_id = payload.get("suspension_id")
        tool_call_id = payload.get("tool_call_id")
        tool_name = payload.get("tool_name")
        assistant_message = payload.get("assistant_message")
        if not isinstance(suspension_id, str) or not suspension_id.strip():
            raise ValueError("suspended checkpoint is missing suspension_id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise ValueError("suspended checkpoint is missing tool_call_id")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("suspended checkpoint is missing tool_name")
        if not isinstance(assistant_message, dict):
            raise ValueError("suspended checkpoint is missing assistant_message")
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("suspended checkpoint metadata must be an object")
        return cls(
            suspension_id=suspension_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=deepcopy(payload.get("arguments")),
            assistant_message=deepcopy(assistant_message),
            metadata=deepcopy(metadata or {}),
        )


class ToolSuspension(Exception):  # noqa: N818 - public control contract
    """Ask an opted-in runner to pause instead of converting this to a tool error."""

    def __init__(self, suspension_id: str, metadata: dict[str, Any] | None = None) -> None:
        if not isinstance(suspension_id, str) or not suspension_id.strip():
            raise ValueError("suspension_id must not be empty")
        self.suspension_id = suspension_id
        self.metadata = deepcopy(metadata or {})
        super().__init__(f"Tool suspended: {suspension_id}")
