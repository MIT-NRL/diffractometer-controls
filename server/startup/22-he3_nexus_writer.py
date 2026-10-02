from server.writers.he3_nexus import HE3DiffractionNXWriter

he3_nexus_writer = None
he3_nexus_writer_subscription = None

if "RE" in globals():
    he3_nexus_writer = HE3DiffractionNXWriter()
    he3_nexus_writer_subscription = RE.subscribe(he3_nexus_writer.receiver)
