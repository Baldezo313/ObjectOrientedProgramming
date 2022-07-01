from __future__ import annotations

import pytest
import sys
import subprocess
import signal
import time
import datetime
import redis
import logging
from pathlib import Path
from typing import Iterator, Any
from unittest.mock import Mock, patch, call, sentinel
from file7 import StatsList
import file7


@pytest.fixture
def valid_stats() -> StatsList:
    return StatsList([1, 2, 2, 3, 3, 4])


def test_mean(valid_stats: StatsList) -> None:
    assert valid_stats.mean() == 2.5


def test_median(valid_stats: StatsList) -> None:
    assert valid_stats.median() == 2.5
    valid_stats.append(4)
    assert valid_stats.median() == 3


def test_mode(valid_stats: StatsList) -> None:
    assert valid_stats.mode() == [2, 3]
    valid_stats.remove(2)
    assert valid_stats.mode() == [3]


@pytest.fixture
def working_directory(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    working = tmp_path / "some_directory"
    working.mkdir()
    source = working / "data.txt"
    source.write_bytes(b"Hello, world!\n")
    checksum = working / "checksum.txt"
    checksum.write_text("data.txt Old_Checksum")
    yield source, checksum
    checksum.unlink()
    source.unlink()


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="requires python3.9 feature"
)
def test_checksum(working_directory: tuple[Path, Path]) -> None:
    source_path, old_checksum_path = working_directory
    file7.checksum(source_path, old_checksum_path)
    backup = old_checksum_path.with_stem(
        f"(old) {old_checksum_path.stem}"
    )
    assert backup.exists()
    assert old_checksum_path.exists()
    name, checksum = old_checksum_path.read_text().rstrip().split()
    assert name == source_path.name
    assert (
        checksum == "d9014c4624844aa5bac314773d6b689a"
        "d467fa4e1d1a50a1b8a99d5f72ff5"
    )


@pytest.fixture(scope="session")
def log_catcher() -> Iterator[None]:
    print("loading server")
    p = subprocess.Popen(
        ["python3", "src/file7.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(0.25)
    yield
    p.terminate()
    p.wait()
    if p.stdout:
        print(p.stdout.read())
    assert (
        p.returncode == -signal.SIGTERM.value
    ), f"Error in watcher, returncode={p.returncode}"


@pytest.fixture
def logging_config() -> Iterator[None]:
    HOST, PORT = "localhost", 18842
    socket_handler = logging.handlers.SocketHandler(HOST, PORT)
    file7.logger.addHandler(socket_handler)
    yield
    socket_handler.close()
    file7.logger.removeHandler(socket_handler)


def test_1(log_catcher: None, logging_config: None) -> None:
    for i in range(10):
        r = file7.work(i)


def test_2(log_catcher: None, logging_config: None) -> None:
    for i in range(1, 10):
        r = file7.work(52 * i)


def test_simple_skip() -> None:
    if sys.platform != "ios":
        pytest.skip("Test works only on Pythonista for ios")
    import location  # type: ignore [import]
    img = location.render_map_snapshot(36.8508, -76.2859)
    assert img is not None


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="requires 3.9, Path.removeprefix()"
)
def test_feature_python39() -> None:
    file_name = "(old) myfile.dat"
    assert file_name.removeprefix("(old) ") == "myfile.dat"


@pytest.fixture
def mock_redis() -> Mock:
    mock_redis_instance = Mock(set=Mock(return_value=True))
    return mock_redis_instance


@pytest.fixture
def tracker(
        monkeypatch: pytest.MonkeyPatch, mock_redis: Mock
) -> file7.FlightStatusTracker:
    fst = file7.FlightStatusTracker()
    monkeypatch.setattr(fst, "redis", mock_redis)
    return fst


def test_monkeypatch_class(
        tracker: file7.FlightStatusTracker, mock_redis: Mock
) -> None:
    with pytest.raises(ValueError) as ex:
        tracker.change_status("AC101", "lost")
    assert ex.value.args[0] == "'lost' is not a valid Status"
    assert mock_redis.set.call_count == 0


def test_patch_class(
        tracker: file7.FlightStatusTracker, mock_redis: Mock
) -> None:
    fake_now = datetime.datetime(2022, 10, 26, 23, 45, 50)
    utc = datetime.timezone.utc
    with patch("file7.datetime") as mock_datetime:
        mock_datetime.datetime = Mock(now=Mock(return_value=fake_now))
        mock_datetime.timezone = Mock(utc=utc)
        tracker.change_status(
            "AC101", file7.Status.ON_TIME
        )
        mock_datetime.datetime.now.assert_called_once_with(tz=utc)
        expected = f"2022-10-26T23:45:50|ON TIME"
        mock_redis.set.assert_called_once_with("flightno:AC101", expected)


@pytest.fixture
def mock_hashlib(monkeypatch) -> Mock:
    mocked_hashlib = Mock(sha256=Mock(return_value=sentinel.checksum))
    monkeypatch.setattr(file7, "hashlib", mocked_hashlib)
    return mocked_hashlib


def test_file_checksum(mock_hashlib, tmp_path) -> None:
    source_file = tmp_path / "some_file"
    source_file.write_text("")
    cw = file7.FileChecksum(source_file)
    assert cw.source == source_file
    assert cw.checksum == sentinel.checksum

    



