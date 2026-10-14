"""Disposable loopback Redis protocol server for demo QueueServer processes."""
import argparse
from fakeredis import TcpFakeServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    with TcpFakeServer(("127.0.0.1", args.port), server_type="redis") as server:
        server.daemon_threads = True
        server.serve_forever()


if __name__ == "__main__":
    main()
