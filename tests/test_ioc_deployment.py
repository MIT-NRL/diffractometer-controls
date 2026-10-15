import tempfile
from pathlib import Path
import unittest

from server.iocs.deployment import network_config, read_env, split_save, stats_aliases


class IOCDeploymentTests(unittest.TestCase):
    def test_autosave_entries_and_arrays_keep_one_owner(self):
        source = '''# header
4dh4:m1.VAL 42
4dh4:he3PSD:NBins 500
4dh4:TIFF1:FileName sample_name
4dh4:USBCTR:MCS:Data @array@ { "1" "2"
 "3" "4" }
4dh4:Bluesky:ActiveMode TEST
<END>
'''
        parts = split_save(source, ("4dh4:TIFF1:",))
        self.assertIn('4dh4:m1.VAL 42', parts['main'])
        self.assertIn('4dh4:Bluesky:ActiveMode TEST', parts['main'])
        self.assertIn('4dh4:he3PSD:NBins 500', parts['psd'])
        self.assertIn('4dh4:TIFF1:FileName sample_name', parts['imaging'])
        self.assertIn('"3" "4" }', parts['daq'])
        for pv in ('4dh4:m1.VAL', '4dh4:he3PSD:NBins', '4dh4:TIFF1:FileName', '4dh4:USBCTR:MCS:Data'):
            self.assertEqual(sum(pv in value for value in parts.values()), 1)
        self.assertTrue(all(p.count('<END>') == 1 for p in parts.values()))

    def test_network_keeps_existing_ports_and_allocates_unique_new_ports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'network.env'
            path.write_text('''MITR_EPICS_CA_ADDR_LIST='host:5067 host:5066'
MITR_EPICS_PVA_ADDR_LIST='host:5074 host:5078'
MITR_EPICS_SERVER_INTERFACE='192.0.2.1'
MITR_MAIN_CA_SERVER_PORT=5067
MITR_MAIN_PVA_SERVER_PORT=5075
MITR_MAIN_PVA_BROADCAST_PORT=5074
REOLINK_CA_SERVER_PORT=5066
REOLINK_PVA_SERVER_PORT=5077
REOLINK_PVA_BROADCAST_PORT=5078
''')
            path.write_text(network_config(path))
            values = read_env(path)
            self.assertEqual(values['MITR_MAIN_CA_SERVER_PORT'], '5067')
            ports = [v for k, v in values.items() if k.endswith('_PORT')]
            self.assertEqual(len(ports), len(set(ports)))
            self.assertEqual(len(values['MITR_EPICS_CA_ADDR_LIST'].split()), 5)
            original = path.read_text()
            self.assertEqual(network_config(path), original)

    def test_stats_aliases_retain_old_names(self):
        db = 'record(ai, "$(IOC):HEARTBEAT") { alias("$(IOC):OLD_HEARTBEAT") }\nrecord(stringin, "$(IOC):UPTIME") {}'
        self.assertEqual(stats_aliases('4dh4:IOC:Motors', '4dh4', db),
                         'alias("4dh4:IOC:Motors:HEARTBEAT", "4dh4:HEARTBEAT")\n'
                         'alias("4dh4:IOC:Motors:OLD_HEARTBEAT", "4dh4:OLD_HEARTBEAT")\n'
                         'alias("4dh4:IOC:Motors:UPTIME", "4dh4:UPTIME")\n')
