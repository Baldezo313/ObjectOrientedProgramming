from __future__ import annotations

import csv
import logging
from contextlib import contextmanager
from functools import wraps, lru_cache
from pprint import pprint
from threading import Timer
from urllib.request import urlopen
import datetime
import math
import random
import pickle
import json
from math import radians, pi, hypot, cos, factorial
from pathlib import Path
import abc
from PIL import Image
from types import TracebackType
from decimal import Decimal
from typing import NamedTuple, Protocol, Sequence, Tuple, Match, Any, Iterable, Iterator, Optional, List, Dict, TextIO, Callable, cast, Type, Literal, Pattern, Match
from urllib.parse import urlparse
import contextlib
import socket
import re
import os
import os.path
import subprocess
import sys
import heapq
import time
import gzip
import io
from dataclasses import dataclass, field


# ================================
source = "hello world"
# pattern = r"hello world"
pattern = re.compile(
    r"(?P<dt>\w\w\w \d\d, \d\d\d\d, \d\d:\d\d:\d\d)"
    r"\s+(?P<level>\w+)"
    r"\s+(?P<msg>.*)"
)

"""
The Iterator Pattern
"""
# =======================================


class CapitalIterable(Iterable[str]):
    def __init__(self, string: str) -> None:
        self.string = string

    def __iter__(self) -> Iterator[str]:
        return CapitalIterator(self.string)


class CapitalIterator(Iterator[str]):
    def __init__(self, string: str) -> None:
        self.words = [w.capitalize() for w in string.split()]
        self.index = 0

    def __next__(self) -> str:
        if self.index == len(self.words):
            raise StopIteration()
        word = self.words[self.index]
        self.index += 1
        return word


iterable = CapitalIterable('the quick brown fox jumps over the lazy dog')
iterator = iter(iterable)
while True:
    try:
        print(next(iterator))
    except StopIteration:
        break


for i in iterable:
    print(i)

input_strings = ["1", "5", "28", "131", "3"]
output_integers = []
for num in input_strings:
    output_integers.append(int(num))

output_integers = [int(num) for num in input_strings]

output_integers = [int(num) for num in input_strings if len(num) < 3]
print(output_integers)

source_path = Path('C:/Users/balde/Desktop/Oreilli') / 'file1.py'
with source_path.open() as source:
    examples = [line.rstrip()
                for line in source
                if ">>>" in line]

with source_path.open() as source:
    examples = [(number, line.rstrip())
                for number, line in enumerate(source, start=1)
                if ">>>" in line]


class Book(NamedTuple):
    author: str
    title: str
    genre: str


books = [
    Book("Pratchett", "Nightwatch", "fantasy"),
    Book("Pratchett", "Thief Of Time", "fantasy"),
    Book("Le Guin", "The Dispossessed", "scifi"),
    Book("Le Guin", "A Wizard Of Earthsea", "fantasy"),
    Book("Jemisin", "The Broken Earth", "fantasy"),
    Book("Turner", "The Thief", "fantasy"),
    Book("Phillips", "Preston Diamond", "western"),
    Book("Phillips", "Twice Upon A Time", "scifi"),
]

fantasy_authors = {b.author for b in books if b.genre == "fantasy"}
print(fantasy_authors)

fantasy_titles = {b.title: b for b in books if b.genre == "fantasy"}
print(fantasy_titles['Nightwatch'])
print(fantasy_titles)


def extract_and_parse_1(
        full_log_path: Path, warning_log_path: Path
) -> None:
    with warning_log_path.open("w") as target:
        writer = csv.writer(target, delimiter="\t")
        pattern = re.compile(
            r"(\w\w\w \d\d, \d\d\d\d \d\d:\d\d:\d\d) (\w+) (.*)"
        )
    with full_log_path.open() as source:
        for line in source:
            if "WARN" in line:
                line_groups = cast(
                    Match[str], pattern.match(line)
                ).groups()
                writer.writerow(line_groups)


class WarningReformat(Iterator[Tuple[str, ...]]):
    pattern = re.compile(
        r"(\w\w\w \d\d, \d\d\d\d \d\d:\d\d:\d\d) (\w+) (.*)"
    )

    def __init__(self, source: TextIO) -> None:
        self.insequence = source

    def __iter__(self) -> Iterator[tuple[str, ...]]:
        return self

    def __next__(self) -> tuple[str, ...]:
        line = self.insequence.readline()
        while line and "WARN" not in line:
            line = self.insequence.readline()
        if not line:
            raise StopIteration
        else:
            return tuple(
                cast(Match[str],
                     self.pattern.match(line)).groups()
            )


def extract_and_parse_2(
        full_log_path: Path, warning_log_path: Path
) -> None:
    with warning_log_path.open("w") as target:
        writer = csv.writer(target, delimiter="\t")
        with full_log_path.open() as source:
            filter_reformat = WarningReformat(source)
            for line_groups in filter_reformat:
                writer.writerow(line_groups)


def warnings_filter(
        source: Iterable[str]
) -> Iterator[tuple[str, ...]]:
    pattern = re.compile(
        r"(\w\w\w \d\d, \d\d\d\d \d\d:\d\d:\d\d) (\w+) (.*)"
    )
    for line in source:
        if "WARN" in line:
            yield tuple(
                cast(Match[str], pattern.match(line)).groups()
            )


def warnings_filter(source: Iterable[str]) -> Iterator[Sequence[str]]:
    pattern = re.compile(
        r"(\w\w\w \d\d, \d\d\d\d \d\d:\d\d:\d\d) (\w+) (.*)"
    )
    for line in source:
        if match := pattern.match(line):
            if "WARN" in match.group(2):
                yield match.groups()


def exctract_and_parse_3(
        full_log_path: Path, warning_log_path: Path
) -> None:
    with warning_log_path.open("w") as target:
        writer = csv.writer(target, delimiter="\t")
        with full_log_path.open() as infile:
            filter = warnings_filter(infile)
            for line_groups in filter:
                writer.writerow(line_groups)


# print(warnings_filter([]))
# warnings_filter = (
#     tuple(cast(Match[str], pattern.match(line)).groups())
#     for line in source
#     if "WARN" in line
# )

def file_extract(
        path_iter: Iterable[Path]
) -> Iterator[tuple[str, ...]]:
    for path in path_iter:
        with path.open() as infile:
            yield from warnings_filter(infile)


def extract_and_parse_d(
        directory: Path, warning_log_path: Path
) -> None:
    with warning_log_path.open("w") as target:
        writer = csv.writer(target, delimiter="\t")
        log_files = list(directory.glob("sample*.log"))
        for line_groups in file_extract(log_files):
            writer.writerow(line_groups)


warnings_filter = (
    tuple(cast(Match[str], pattern.match(line)).groups())
    for line in source
    if "WARN" in line
)


possible_match_iter = (pattern.match(line) for line in source)
group_iter = (match.groups() for match in possible_match_iter if match)
warnings_filter = (group for group in group_iter if "WARN" in group[1])

possible_match_iter = (pattern.match(line) for line in source)
group_iter = (match.groupdict() for match in possible_match_iter if match)
warning_iter = (
    group for group in group_iter if "WARN" in group["level"]
)
dt_iter = (
    (
        datetime.datetime.strptime(g["dt"], "%b" "%d", "%Y %H:%M:%S"),
        g["level"],
        g["msg"],
    )
    for g in warning_iter
)
warnings_filter = (
    (g[0].isoformat(), g[1], g[2]) for g in dt_iter
)

possible_match_iter = map(pattern.match, source)
good_match_iter = filter(None, possible_match_iter)
group_iter = map(lambda m: m.groupdict(), good_match_iter)
warnings_iter = filter(lambda g: "WARN" in g["level"], group_iter)
dt_iter = map(
    lambda g: (
        datetime.datetime.strptime(g["dt"], "%b" "%d", "%Y %H:%M:%S"),
        g["level"],
        g["msg"],
    ),
    warnings_iter,
)
warnings_filter = map(
    lambda g: (g[0].isoformat(), g[1], g[2]), dt_iter
)


# ==============================================================================================================
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
COMMON DESIGN PATTERNS
"""
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ============================================================================================================


def main_1() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 2401))
    server.listen(1)
    with contextlib.closing(server):
        while True:
            client, addr = server.accept()
            dice_response(client)
            client.close()


def dice_response(client: socket.socket) -> None:
    request = client.recv(1024)
    try:
        response = dice_roller(request)
    except (ValueError, KeyError) as ex:
        response = repr(ex).encode("utf-8")
    client.send(response)


def dice_roller(request: bytes) -> bytes:
    request_text = request.decode("utf-8")
    numbers = [random.randint(1, 6) for _ in range(6)]
    response = f"{request_text} = {numbers}"
    return response.encode("utf-8")


def main() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.connect(("localhost", 2401))
    count = input("How many rolls: ") or "1"
    pattern = input("Dice pattern nd6[dk+-]a: ") or "d6"
    command = f"Dice {count} {pattern}"
    server.send(command.encode("utf-8"))
    server.close()


dice_pattern = re.compile(r"(?P<n>\d*)d(?P<d>\d+)(?P<a>[dk+-]\d+)*")

# if __name__ == "__main__":
#     main()


class LogSocket:
    def __init__(self, socket: socket.socket) -> None:
        self.socket = socket

    def recv(self, count: int = 0) -> bytes:
        data = self.socket.recv(count)
        print(
            f"Receiving {data!r} from {self.socket.getpeername()[0]}"
        )
        return data

    def send(self, data: bytes) -> None:
        print(f"Sending {data!r} to {self.socket.getpeername()[0]}")
        self.socket.send(data)

    def close(self) -> None:
        self.socket.close()


def main_2() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 2401))
    server.listen(1)
    with contextlib.closing(server):
        while True:
            client, addr = server.accept()
            logging_socket = cast(socket.socket, LogSocket(client))
            dice_response(logging_socket)
            client.close()


Address = Tuple[str, int]


class LogRoller:
    def __init__(
            self,
            dice: Callable[[bytes], bytes],
            remote_addr: Address
    ) -> None:
        self.dice_roller = dice
        self.remote_addr = remote_addr

    def __call__(self, request: bytes) -> bytes:
        print(f"Receivng {request!r} from {self.remote_addr}")
        dice_roller = self.dice_roller
        response = dice_roller(request)
        print(f"Sending {response!r} to {self.remote_addr}")
        return response


class ZipRoller:
    def __init__(self, dice: Callable[[bytes], bytes]) -> None:
        self.dice_roller = dice

    def __call__(self, request: bytes) -> bytes:
        dice_roller = self.dice_roller
        response = dice_roller(request)
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="w") as zipfile:
            zipfile.write(response)
        return buffer.getvalue()


def dice_response_bis(client: socket.socket) -> None:
    request = client.recv(1024)
    try:
        remote_addr = client.getpeername()
        roller_1 = ZipRoller(dice_roller)
        roller_2 = LogRoller(roller_1, remote_addr=remote_addr)
        response = roller_2(request)
    except (ValueError, KeyError) as ex:
        response = repr(ex).encode("utf-8")
    client.send(response)


def log_args(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped_function(*args: Any, **kwargs: Any) -> Any:
        print(f"Calling {function.__name__} (*{args}, **{kwargs})")
        result = function(*args, **kwargs)
        return result
    return wrapped_function


# def test1(a: int, b: int, c: int) -> float:
#     return sum(range(a, b + 1)) / c
#
#
# test1 = log_args(test1)
# print(test1(1, 9, 2))


@log_args
def test1(a: int, b: int, c: int) -> float:
    return sum(range(a, b + 1)) / c


# def binom(n: int, k: int) -> int:
#     return factorial(n) // (factorial(k) * factorial(n - k))
#
#
# print(f"6-card deals: {binom(52, 6):,d}")


@lru_cache(64)
def binom(n: int, k: int) -> int:
    return factorial(n) // (factorial(k) * factorial(n-k))


# print(f"6-card deals: {binom(52, 6):,d}")


class NamedLogger:
    def __init__(self, logger_name: str) -> None:
        self.logger = logging.getLogger(logger_name)

    def __call__(
            self,
            function: Callable[..., Any]
    ) -> Callable[..., Any]:
        @wraps(function)
        def wrapped_function(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = function(*args, **kwargs)
                us = (time.perf_counter() - start) * 100_000_000
                self.logger.info(
                    f"{function.__name__}, { us:.1f}us"
                )
                return result
            except Exception as ex:
                us = (time.perf_counter() - start) * 100_000_000
                self.logger.error(
                    f"{ex}, {function.__name__}, { us:.1f}us"
                )
                raise
        return wrapped_function


@NamedLogger("log4")
def test4(median: float, sample: float) -> float:
    return abs(sample - median)


class Observer(Protocol):
    def __call__(self) -> None:
        ...


class Observable:
    def __init__(self) -> None:
        self._observers: list[Observer] = []

    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def _notify_observers(self) -> None:
        for observer in self._observers:
            observer()


Hand = List[int]


class Adjustment:
    def __init__(self, amount: int) -> None:
        self.amount = amount

    @abc.abstractmethod
    def apply(self, dice: "Dice") -> None:
        ...


class Roll(Adjustment):
    def __init__(self, n: int, d: int) -> None:
        self.n = n
        self.d = d

    def apply(self, dice: "Dice") -> None:
        dice.dice = sorted(
            random.randint(1, self.d) for _ in range(self.n)
        )
        dice.modifier = 0


class Drop(Adjustment):
    def apply(self, dice: "Dice") -> None:
        dice.dice = dice.dice[self.amount:]


class Keep(Adjustment):
    def apply(self, dice: "Dice") -> None:
        dice.dice = dice.dice[: self.amount]


class Plus(Adjustment):
    def apply(self, dice: "Dice") -> None:
        dice.modifier += self.amount


class Minus(Adjustment):
    def apply(self, dice: "Dice") -> None:
        dice.modifier -= self.amount


class Dice:
    def __init__(self, n: int, d: int, *adj: Adjustment) -> None:
        self.adjustments = [cast(Adjustment, Roll(n, d))] + list(adj)
        self.dice = list[int]
        self.modifier: int

    def roll(self) -> int:
        for a in self.adjustments:
            a.apply(self)
        return sum(self.dice) + self.modifier

    @classmethod
    def from_text(cls, dice_text: str) -> "Dice":
        dice_pattern = re.compile(
            r"(?P<n>\d*)d(?P<d>\d+)(?P<a>[dk+-]\d+)*"
        )
        adjustement_pattern = re.compile(r"([dk+-])(\d+)")
        adj_class: dict[str, Type[Adjustment]] = {
            "d": Drop,
            "k": Keep,
            "+": Plus,
            "-": Minus,
        }
        if (dice_match := dice_pattern.match(dice_text)) is None:
            raise ValueError(f"Error in {dice_text!r}")
        n = int(dice_match.group("n")) if dice_match.group("n") else 1
        d = int(dice_match.group("d"))
        adjustement_matches = adjustement_pattern.finditer(
            dice_match.group("a") or ""
        )
        adjustements = [
            adj_class[a.group(1)](int(a.group(2)))
            for a in adjustement_matches
        ]
        return cls(n, d, *adjustements)


class ZonkHandHistory(Observable):
    def __init__(self, player: str, dice_set: Dice) -> None:
        super().__init__()
        self.player = player
        self.dice_set = dice_set
        self.rolls: list[Hand]

    def start(self) -> Hand:
        self.dice_set.roll()
        self.rolls = [self.dice_set.dice]
        self._notify_observers()     #state change
        return self.dice_set.dice

    def roll(self) -> Hand:
        self.dice_set.roll()
        self.rolls.append(self.dice_set.dice)
        self._notify_observers()     # state change
        return self.dice_set.dice


class SaveZonkHand(Observer):
    def __init__(self, hand: ZonkHandHistory) -> None:
        self.hand = hand
        self.count = 0

    def __call__(self) -> None:
        self.count += 1
        message = {
            "player": self.hand.player,
            "sequence": self.count,
            "hands": json.dumps(self.hand.rolls),
            "time": time.time(),
        }
        print(f"SaveZonkHand {message}")


# d = Dice.from_text("6d6")
# player = ZonkHandHistory("Bo", d)
# save_history = SaveZonkHand(player)
# player.attach(save_history)
# r1 = player.start()
# print(r1)
# r2 = player.roll()
# print(r2)


class ThreePairZonkHand:
    """Observer of ZonkHandHistory"""
    def __init__(self, hand: ZonkHandHistory) -> None:
        self.hand = hand
        self.zonked = False

    def __call__(self) -> None:
        last_roll = self.hand.rolls[-1]
        distinct_values = set(last_roll)
        self.zonked = len(distinct_values) == 3 and all(
            last_roll.count(v) == 2 for v in distinct_values
        )
        if self.zonked:
            print("3 Pair Zonk!")


Size = Tuple[int, int]


class FillAlgorithm(abc.ABC):
    @abc.abstractmethod
    def make_background(
            self,
            img_file: Path,
            desktop_size: Size
    ) -> Image:
        pass


class TiledStrategy(FillAlgorithm):
    def make_background(
            self,
            img_file: Path,
            desktop_size: Size
    ) -> Image:
        in_img = Image.open(img_file)
        out_img = Image.new("RGB", desktop_size)
        num_titles = [
            o // i + 1 for o, i in zip(out_img.size, in_img.size)
        ]
        for x in range(num_titles[0]):
            for y in range(num_titles[1]):
                out_img.paste(
                    in_img,
                    (
                        in_img.size[0] * x,
                        in_img.size[1] * y,
                        in_img.size[0] * (x + 1),
                        in_img.size[1] * (y + 1),
                    ),
                )
        return out_img


class CenteredStrategy(FillAlgorithm):
    def make_background(
            self,
            img_file: Path,
            desktop_size: Size
    ) -> Image:
        in_img = Image.open(img_file)
        out_img = Image.new("RGB", desktop_size)
        left = (out_img.size[0] - in_img.size[0]) // 2
        top = (out_img.size[1] - in_img.size[1]) // 2
        out_img.paste(
            in_img,
            (left, top, left + in_img.size[0], top + in_img.size[1]),
        )
        return out_img


class ScaledStrategy(FillAlgorithm):
    def make_background(
            self,
            img_file: Path,
            desktop_size: Size
    ) -> Image:
        in_img = Image.open(img_file)
        out_img = in_img.resize(desktop_size)
        return out_img


class Resizer:
    def __init__(self, algorithm: FillAlgorithm) -> None:
        self.algorithm = algorithm

    def resize(self, image_file: Path, size: Size) -> Image:
        result = self.algorithm.make_background(image_file, size)
        return result


def main_img() -> None:
    image_file = Path.cwd() / "boat.png"
    tiled_desktop = Resizer(TiledStrategy())
    tiled_image = tiled_desktop.resize(image_file, (1920, 1080))
    tiled_image.show()


class NMEA_State:
    def __init__(self, message: "Message") -> None:
        self.message = message

    def feed_byte(self, input: int) -> "NMEA_State":
        return self

    def valid(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message})"


class Message:
    def __init__(self) -> None:
        self.body = bytearray(80)
        self.checksum_source = bytearray(2)
        self.body_len = 0
        self.checksum_len = 0
        self.checksum_computed = 0

    def reset(self) -> None:
        self.body_len = 0
        self.checksum_len = 0
        self.checksum_computed = 0

    def body_append(self, input: int) -> int:
        self.body[self.body_len] = input
        self.body_len += 1
        self.checksum_computed ^= input
        return self.body_len

    def checksum_append(self, input: int) -> int:
        self.checksum_source[self.checksum_len] = input
        self.checksum_len += 1
        return self.checksum_len

    @property
    def valid(self) -> bool:
        return (
            self.checksum_len == 2
            and int(self.checksum_source, 16) == self.checksum_computed
        )


class Header(NMEA_State):
    def __init__(self, message: "Message") -> None:
        self.message = message
        self.message.reset()

    def feed_byte(self, input: int) -> "NMEA_State":
        if input == ord(b"$"):
            return Header(self.message)
        size = self.message.body_append(input)
        if size == 5:
            return Body(self.message)
        return self


class Body(NMEA_State):
    def feed_byte(self, input: int) -> "NMEA_State":
        if input == ord(b"$"):
            return Header(self.message)
        if input == ord(b"*"):
            return Checksum(self.message)
        self.message.body_append(input)
        return self


class Checksum(NMEA_State):
    def feed_byte(self, input: int) -> "NMEA_State":
        if input == ord(b"$"):
            return Header(self.message)
        if input in {ord(b"\n"), ord(b"\r")}:
            # Incomplete checksum ... Will be invalid
            return End(self.message)
        size = self.message.checksum_append(input)
        if size == 2:
            return End(self.message)
        return self


class End(NMEA_State):
    def feed_byte(self, input: int) -> "NMEA_State":
        if input == ord(b"$"):
            return Header(self.message)
        elif input not in {ord(b"\n"), ord(b"\r")}:
            return Waiting(self.message)
        return self

    def valid(self) -> bool:
        return self.message.valid


class Waiting(NMEA_State):
    def feed_byte(self, input: int) -> "NMEA_State":
        if input == ord(b"$"):
            return Header(self.message)
        return self


class Reader:
    def __init__(self) -> None:
        self.buffer = Message()
        self.state: NMEA_State = Waiting(self.buffer)

    def read(self, source: Iterable[bytes]) -> Iterator[Message]:
        for byte in source:
            self.state = self.state.feed_byte(cast(int, byte))
            if self.buffer.valid:
                yield self.buffer
                self.buffer = Message()
                self.state = Waiting(self.buffer)


message = b'''
$GPGGA,161229.487,3723.2475,N,12158.3416,W,1,07,1.0,9.0,M,,,,0000*18
$GPGLL,3723.2475,N,12158.3416,W,161229.487,A,A*41
'''
rdr = Reader()
result = list(rdr.read(message))


class OneOnly:
    _singleton = None

    def __new__(cls, *args, **kwargs):
        if not cls._singleton:
            cls._singleton = super().__new__(cls, *args, **kwargs)
        return cls._singleton


o1 = OneOnly()
o2 = OneOnly()
# print(o1 == o2)
# print(id(o1) == id(o2))
# print(o1)
# print(o2)


class NMEA_Statebis:
    def enter(self, message: "Message") -> "NMEA_Statebis":
        return self

    def feed_byte(
            self,
            message: "Message",
            input: int
    ) -> "NMEA_Statebis":
        return self

    def valid(self, message: "Message") -> bool:
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Waitingbis(NMEA_Statebis):
    def feed_byte(
            self,
            message: "Message",
            input: int
    ) -> "NMEA_Statebis":
        return self
        if input == ord(b"$"):
            return HEADER
        return self


class Headerbis(NMEA_Statebis):
    def enter(self, message: "Message") -> "NMEA_Statebis":
        message.reset()
        return self

    def feed_byte(
            self,
            message: "Message",
            input: int
    ) -> "NMEA_Statebis":
        return self
        if input == ord(b"$"):
            return HEADER
        size = message.body_append(input)
        if size == 5:
            return BODY
        return self


class Bodybis(NMEA_Statebis):
    def feed_byte(
            self,
            message: "Message",
            input: int
    ) -> "NMEA_Statebis":
        return self
        if input == ord(b"$"):
            return Headerbis
        if input == ord(b"*"):
            return CHECKSUM
        size = message.body_append(input)
        return self


class Checksumbis(NMEA_Statebis):
    def feed_byte(
            self,
            message: "Message",
            input: int
    ) -> "NMEA_Statebis":
        return self
        if input == ord(b"$"):
            return HEADER
        if input in {ord(b"\n"), ord(b"\r")}:
            return END
        size = message.checksum_append(input)
        if size == 2:
            return END
        return self


class Endbis(NMEA_Statebis):
    def feed_byte(
            self,
            message: "Message",
            input: int
    ) -> "NMEA_Statebis":
        return self
        if input == ord(b"$"):
            return Headerbis
        elif input not in {ord(b"\n"), ord(b"\r")}:
            return WAITING
        return self

    def valid(self, message: "Message") -> bool:
        return message.valid


WAITING = Waitingbis()
HEADER = Headerbis()
BODY = Bodybis()
CHECKSUM = Checksumbis()
END = Endbis()


class Readerbis:
    def __init__(self) -> None:
        self.buffer = Message()
        self.state: NMEA_Statebis = WAITING

    def read(self, source: Iterable[bytes]) -> Iterator[Message]:
        for byte in source:
            new_state = self.state.feed_byte(
                self.buffer, cast(int, byte)
            )
            if self.buffer.valid:
                yield self.buffer
                self.buffer = Message()
                new_state = WAITING
            if new_state != self.state:
                new_state.enter(self.buffer)
                self.state = new_state



