"""Explicit device namespace and runtime dependencies for startup factories."""

from dataclasses import dataclass, field
from typing import Any, MutableMapping, Mapping


@dataclass
class StartupContext:
    namespace: MutableMapping[str, Any]
    runtime: Mapping[str, Any] = field(default_factory=dict)

    def optional(self, name):
        return self.runtime.get(name, self.namespace.get(name))

    def publish(self, definitions):
        self.namespace.update(definitions)
        return definitions
