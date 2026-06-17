"""Safety and security profiles for the harness runtime."""

from __future__ import annotations

from dataclasses import dataclass
from os import path as os_path
from pathlib import Path
from re import Pattern


class HarnessError(RuntimeError):
    """Base class for harness-specific errors."""


class ToolPermissionError(HarnessError):
    """Raised when a tool execution is not permitted by the security profile."""


@dataclass(frozen=True, slots=True)
class SecurityProfile:
    """Immutable security constraints for a trajectory run.

    The profile is enforced at two boundaries:
    1. Action validation (before an action is even planned as an event).
    2. Tool execution (before the ToolWorker dispatches to the actual tool).
    """

    allowed_tool_names: frozenset[str] = frozenset()
    allowed_workspace_root: Path | None = None
    forbidden_path_pattern: Pattern[str] | None = None
    max_tool_timeout_seconds: float = 30.0

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Return True if ``tool_name`` is in the allow-list.

        An empty allow-list blocks everything (deny-by-default).
        """
        return tool_name in self.allowed_tool_names

    def is_path_allowed(self, path: str) -> bool:
        """Return True if ``path`` is inside the allowed workspace jail.

        When ``allowed_workspace_root`` is unset, the legacy regex guard is
        still applied for backwards compatibility. When a workspace root is
        configured, paths are expanded, resolved through symlinks, and rejected
        if they escape the root. The optional regex remains an additional
        deny-list after path jail validation.
        """
        if self.allowed_workspace_root is not None:
            root = self.allowed_workspace_root.expanduser().resolve()
            target = Path(path).expanduser().resolve()
            if os_path.commonpath([str(root), str(target)]) != str(root):
                return False

        if self.forbidden_path_pattern is None:
            return True
        return self.forbidden_path_pattern.search(path) is None
