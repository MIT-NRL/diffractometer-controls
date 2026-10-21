from server.writers.scalar import ScalarDiffractionWriter

scalar_data_writer = None
scalar_data_writer_subscription = None

if "RE" in globals():
    run_storage.add_writer(ScalarDiffractionWriter)
    scalar_data_writer = ScalarDiffractionWriter
    scalar_data_writer_subscription = run_storage_subscription
