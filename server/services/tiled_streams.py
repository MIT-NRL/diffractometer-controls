"""Constrain Tiled's standard HDF5 adapter to advertised, committed rows."""

from bluesky_tiled_plugins.writing.consolidators import (
    CONSOLIDATOR_REGISTRY, HDF5Consolidator,
)


class CommittedHDF5Consolidator(HDF5Consolidator):
    def adapter_parameters(self):
        parameters = super().adapter_parameters()
        if self._sres_parameters.get("committed_prefix"):
            # The file may already contain the next point, or continue growing
            # during a Tiled outage. Tiled's ArrayAdapter expects exactly its
            # catalogued shape, even when the client requests a smaller slice.
            parameters["slice"] = f"0:{self.shape[0]}"
        return parameters


def install_committed_hdf5_consolidator():
    """Use the documented consolidator registry; leave other resources intact."""
    current = CONSOLIDATOR_REGISTRY["application/x-hdf5"]
    if current is CommittedHDF5Consolidator:
        return
    if current is not HDF5Consolidator:
        raise RuntimeError("An alternate HDF5 consolidator is already installed; configure its committed-prefix support")
    CONSOLIDATOR_REGISTRY["application/x-hdf5"] = CommittedHDF5Consolidator
