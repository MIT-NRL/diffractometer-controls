# diffractometer-controls

## Disconnected demo

From the `diffractometer_controls` directory, run:

```bash
python launcher.py --demo
```

The launcher owns a loopback-only caproto IOC, nonpersistent Redis instance,
Queue Server, and Bluesky document proxy. Closing the GUI stops those child
processes. Demo runs do not connect to Tiled and do not write detector files.
