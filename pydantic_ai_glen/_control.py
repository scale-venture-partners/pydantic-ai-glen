"""Glen's account-level mode, which the Claude Code plugin obeys on every hook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MODES = ("on", "off", "silent", "incognito")

# How long a control status is trusted. The plugin asks once per session; a long-lived agent
# process asks about once a minute so a mode change reaches it.
TTL = 60.0
# A failed lookup is not retried at once, so an unreachable Glen costs one timeout per this long.
FAILURE_TTL = 30.0
TIMEOUT = 3.0


@dataclass(frozen=True)
class Control:
    """What the account allows right now.

    `on` recalls and records. `incognito` recalls but records nothing. `silent` records but
    surfaces nothing. `off` does neither.
    """

    mode: str = "on"
    skill_use: bool = True

    @property
    def recalls(self) -> bool:
        return self.mode in ("on", "incognito")

    @property
    def records(self) -> bool:
        return self.mode in ("on", "silent")


def parse_control(data: Any) -> Control | None:
    """A control status response, or None when it is not one."""
    if not isinstance(data, dict):
        return None
    mode = mode_of(data)
    if mode is None:
        return None
    return Control(mode=mode, skill_use=data.get("skillUse") is not False)


def mode_of(data: dict[str, Any]) -> str | None:
    """The mode a control status or a turn response asks for, the way the plugin reads it."""
    for key in ("serverMode", "mode"):
        if data.get(key) in MODES:
            return str(data[key])
    incognito = data.get("serverIncognito")
    if not isinstance(incognito, bool):
        return None
    if incognito:
        return "incognito"
    return "silent" if data.get("serverSilent") is True else "on"
