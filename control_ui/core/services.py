"""Transports are created and owned by the host application."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ControlServices:
    re_client: Any
    documents: Any
    re_manager_api: Any
