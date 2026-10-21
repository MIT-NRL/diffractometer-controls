"""Instrument import path for the reusable diffraction history loader."""

from diffraction_analysis.history import (
    DIFFRACTION_EXPERIMENT_TYPE,
    _coerce_number,
    _catalog_exists,
    _resolve_catalog_name,
    _run_metadata,
    _run_start_stop_docs,
    _is_diffraction_start_doc,
    _run_data_type,
    _start_live_plot_fields,
    _data_keys_have_plot_payload,
    _legacy_scalar_fields,
    _format_run_timestamp,
    _make_history_entry,
    _plain_mapping,
    _to_plain_value,
    _stream_descriptor_docs,
    _stream_row_count,
    _dataset_column_to_event_list,
    _run_streams,
    _restore_array,
    select_history_stream,
    suggest_diffraction_start_updates,
    build_history_snapshot,
    fetch_recent_history_entries,
)


def _preferred_catalog_names_for_mode(mode):
    if str(mode or "").strip().lower() == "test":
        return ["testdb"]
    return ["4dh4_diffraction", "4dh4", "4dh4Collection"]
