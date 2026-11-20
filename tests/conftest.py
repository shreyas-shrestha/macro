import socket

import pytest


@pytest.fixture
def no_network(monkeypatch):
    """Fail any test that opens a network connection."""

    def connect(*args):
        raise AssertionError("tests must not hit the network")

    monkeypatch.setattr(socket.socket, "connect", connect)
