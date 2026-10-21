from server.writers.he3_nexus import HE3DiffractionNXWriter

he3_nexus_writer = None
he3_nexus_writer_subscription = None

if "RE" in globals():
    run_storage.add_writer(HE3DiffractionNXWriter)
    he3_nexus_writer = HE3DiffractionNXWriter
    he3_nexus_writer_subscription = run_storage_subscription
