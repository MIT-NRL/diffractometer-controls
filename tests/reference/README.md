# Server contract baseline

`server_contracts.json` captures 22 public plans from original startup revision
`806496621ba808f7c9c5994030f71788a3350cb1`, using simulated/fake devices and mocked
services, before definition extraction. Source was read as UTF-8.

Golden values include names, generator status, signatures/defaults, QueueServer
annotations and docstrings. `tests.test_server_startup` compares new ordered
startup against them, runs all 15 numbered files with fake hardware/services,
and guards definition imports against EPICS construction/read/write calls.

Existing scalar, PSD, simulation and imaging tests cover scientific behavior.
Actual writer suites and hardware acceptance remain required on Linux; mocked
startup does not prove writer output compatibility.
