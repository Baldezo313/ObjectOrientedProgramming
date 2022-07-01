from __future__ import annotations

import collections
import json
import tarfile
import socketserver
import hashlib
import pickle
import struct
import logging
import logging.handlers
import time
import datetime
import sys
import redis
from enum import Enum
from math import factorial
from pathlib import Path
from typing import Optional, Any, Callable, List, DefaultDict, TextIO

import unittest


# ======================================================================================================================
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
TESTING OBJECT-ORIENTED PROGRAMS
"""
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ======================================================================================================================


def average(data: list[Optional[int]]) -> float:
    """
    GIVEN a list, data = [1, 2, None, 3, 4]
    WHEN we compute m = average(data)
    THEN the result, m, is 2.5
    """
    pass


class CheckNumbers(unittest.TestCase):
    def test_int_float(self) -> None:
        self.assertEqual(1, 1.0)

    def test_str_float(self) -> None:
        self.assertEqual(1, "1")


if __name__ == "__main__":
    unittest.main()


class TestNumbers:
    def test_int_float(self) -> None:
        assert 1 == 1.0

    def test_int_str(self) -> None:
        assert 1 == "1"


def setup_module(module: Any) -> None:
    print(f"setting up MODULE {module.__name__}")


def teardown_module(module: Any) -> None:
    print(f"tearing down MODULE {module.__name__}")


def test_a_function() -> None:
    print("RUNNING TEST FUNCTION")


class BaseTest:
    @classmethod
    def setup_class(cls: type["BaseTest"]) -> None:
        print(f"setting up CLASS {cls.__name__}")

    @classmethod
    def teardown_class(cls: type["BaseTest"]) -> None:
        print(f"tearing down CLASS {cls.__name__}\n")

    def setup_method(self, method: Callable[[], None]) -> None:
        print(f"setting up METHOD {method.__name__}")

    def teardown_method(self, method: Callable[[], None]) -> None:
        print(f"tearing down METHOD {method.__name__}")


class TestClass1(BaseTest):
    def test_method_1(self) -> None:
        print(f"RUNNING METHOD 1-1")

    def test_method_2(self) -> None:
        print(f"RUNNING METHOD 1-2")


class TestClass2(BaseTest):
    def test_method_1(self) -> None:
        print("RUNNING METHOD 2-1")

    def test_method_2(self) -> None:
        print("RUNNING METHOD 2-2")


class StatsList(List[Optional[float]]):
    """Stats with None objects rejected"""
    def mean(self) -> float:
        clean = list(filter(None, self))
        return sum(clean) / len(clean)

    def median(self) -> float:
        clean = list(filter(None, self))
        if len(clean) % 2:
            return clean[len(clean) // 2]
        else:
            idx = len(clean) // 2
            return (clean[idx] + clean[idx - 1]) / 2

    def mode(self) -> list[float]:
        freqs: DefaultDict[float, int] = collections.defaultdict(int)
        for item in filter(None, self):
            freqs[item] += 1
        mode_freq = max(freqs.values())
        modes = [item
                 for item, value in freqs.items()
                 if value == mode_freq]
        return modes


def checksum(source: Path, checksum_path: Path) -> None:
    if checksum_path.exists():
        backup = checksum_path.with_stem(f"(old) {checksum_path.stem}")
        backup.write_text(checksum_path.read_text())
    checksum = hashlib.sha256(source.read_bytes())
    checksum_path.write_text(f"{source.name} {checksum.hexdigest()}\n")


class LogDataCatcher(socketserver.BaseRequestHandler):
    log_file: TextIO
    count: int = 0
    size_format = ">L"
    size_bytes = struct.calcsize(size_format)

    def handle(self) -> None:
        size_header_bytes = self.request.recv(LogDataCatcher.size_bytes)
        while size_header_bytes:
            payload_size = struct.unpack(
                LogDataCatcher.size_format, size_header_bytes
            )
            payload_bytes = self.request.recv(payload_size[0])
            payload = pickle.loads(payload_bytes)
            LogDataCatcher.count += 1
            self.log_file.write(json.dumps(payload) + "\n")
            try:
                size_header = self.request.recv(
                    LogDataCatcher.size_bytes
                )
            except (ConnectionResetError, BrokenPipeError):
                break


def main(host: str, port: int, target: Path) -> None:
    with target.open("w") as unified_log:
        LogDataCatcher.log_file = unified_log
        with socketserver.TCPServer(
                (host, port), LogDataCatcher
        ) as server:
            server.serve_forever()


# if __name__ == "__main__":
#     HOST, PORT = "localhost", 18842
#     main(HOST, PORT, Path("one.log"))


logger = logging.getLogger("app")


def work(i: int) -> int:
    logger.info("Factorial %d", i)
    f = factorial(i)
    logger.info("Factorial(%d) = %d", i, f)
    return f


# if __name__ == "__main__":
#     HOST, PORT = "localhost", 18842
#     socket_handler = logging.handlers.SocketHandler(HOST, PORT)
#     stream_handler = logging.StreamHandler(sys.stderr)
#     logging.basicConfig(
#         handlers=[socket_handler, stream_handler],
#         level=logging.INFO
#     )
#     for i in range(10):
#         work(i)
#     logging.shutdown()


class Status(str, Enum):
    CANCELLED = "CANCELLED"
    DELAYED = "DELAYED"
    ON_TIME = "ON TIME"


class FlightStatusTracker:
    def __init__(
            self,
            redis_instance: Optional[redis.Connection] = None
    ) -> None:
        self.redis = (
            redis_instance
            if redis_instance
            else redis.Redis(host="127.0.0.1", port=6379, db=0)
        )

    def change_status(self, flight: str, status: Status) -> None:
        if not isinstance(status, Status):
            raise ValueError(f"{status!r} is not a valid Status")
        key = f"flightno:{flight}"
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        value = f"{now.isoformat()}|{status.value}"
        self.redis.set(key, value)

    def get_status(self, flight: str) -> tuple[datetime.datetime, Status]:
        key = f"flightno:{flight}"
        value = self.redis.get(key).decode("utf-8")
        text_timestamp, text_status = value.split("|")
        timestamp = datetime.datetime.fromisoformat(text_timestamp)
        status = Status(text_status)
        return timestamp, status


class FileChecksum:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.checksum = hashlib.sha256(source.read_bytes())
