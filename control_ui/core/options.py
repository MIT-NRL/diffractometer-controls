"""Host-owned options for custom plan widgets; no site or transport creation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanEditorOptions:
    query_mode: str = "local"
    cache_ttl_s: float = 2.0
    stream_address: str = ""
    snapshot_address: str = ""
    stream_topic: str = "file_dir_choices"
    local_roots: tuple[str, ...] = ()
    max_depth: int = 3
    worker_function: str = "list_imaging_file_dirs"
