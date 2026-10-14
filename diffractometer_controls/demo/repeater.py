"""Loopback-only Channel Access repeater on the dedicated demo port."""
from caproto.sync.repeater import run

if __name__ == "__main__":
    run(host="127.0.0.1")
