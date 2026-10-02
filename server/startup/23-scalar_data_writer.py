from server.writers.scalar import ScalarDiffractionWriter

scalar_data_writer = None
scalar_data_writer_subscription = None

if "RE" in globals():
    scalar_data_writer = ScalarDiffractionWriter()
    scalar_data_writer_subscription = RE.subscribe(scalar_data_writer.receiver)
