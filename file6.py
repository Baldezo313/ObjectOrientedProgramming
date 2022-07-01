from __future__ import annotations

import csv
import logging
import weakref
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
import subprocess
from math import radians, pi, hypot, cos, factorial
from pathlib import Path
import abc
from PIL import Image
from types import TracebackType
from decimal import Decimal
from typing import NamedTuple, Protocol, Sequence, Tuple, Match, Any, Iterable, Iterator, Optional, List, Dict, TextIO, \
    Callable, cast, Type, Literal, Pattern, Match, overload, Union, ContextManager
from urllib.parse import urlparse
from enum import Enum, auto
import contextlib
import sqlite3
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


class TimeSince:
    """Expects time as six digits, no punctuation."""

    def parse_time(self, time: str) -> tuple[float, float, float]:
        return (
            float(time[0:2]),
            float(time[2:4]),
            float(time[4:]),
        )

    def __init__(self, starting_time: str) -> None:
        self.hr, self.min, self.sec = self.parse_time(starting_time)
        self.start_seconds = ((self.hr * 60) + self.min) * 60 + self.sec

    def interval(self, log_time: str) -> float:
        log_hr, log_min, log_sec = self.parse_time(log_time)
        log_seconds = ((log_hr * 60) + log_min) * 60 + log_sec
        return log_seconds - self.start_seconds


ts = TimeSince("000123")  # Log started at 00:01:23
# print(ts.interval("020304"))
# print(ts.interval("030405"))

data = [
    ("000123", "INFO", "Gila Flats 1959-08-20"),
    ("000142", "INFO", "test block 15"),
    ("004201", "ERROR", "intrinsic field chamber door locked"),
    ("004210.11", "INFO", "generator power active"),
    ("004232.33", "WARNING", "extra mass detected")
]


# class LogProcessor:
#     def __init__(self, log_entries: list[tuple[str, str, str]]) -> None:
#         self.log_entries = log_entries
#
#     def report(self) -> None:
#         first_time, first_sev, first_msg = self.log_entries[0]
#         for log_time, severity, message in self.log_entries:
#             if severity == "ERROR":
#                 first_time = log_time
#             # interval = ??? Need to compute an interval ???
#             print(f"{interval:8.2f} | {severity:7s} {message}")


class IntervalAdapter:
    def __init__(self) -> None:
        self.ts: Optional[TimeSince] = None

    def time_offset(self, start: str, now: str) -> float:
        if self.ts is None:
            self.ts = TimeSince(start)
        else:
            h_m_s = self.ts.parse_time(start)
            if h_m_s != (self.ts.hr, self.ts.min, self.ts.sec):
                self.ts = TimeSince(start)
        return self.ts.interval(now)


class LogProcessor:
    def __init__(
            self,
            log_entries: list[tuple[str, str, str]]
    ) -> None:
        self.log_entries = log_entries
        self.time_convert = IntervalAdapter()

    def report(self) -> None:
        first_time, first_sev, first_msg = self.log_entries[0]
        for log_time, severity, message in self.log_entries:
            if severity == "ERROR":
                first_time = log_time
                interval = self.time_convert.time_offset(first_time, log_time)
                print(f"{interval:8.2f} | {severity:7s} {message}")


class FindUML:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.start_pattern = re.compile(r"@startuml *(.*)")

    def uml_file_iter(self) -> Iterator[tuple[Path, Path]]:
        for source in self.base.glob("***/*.uml"):
            if any(n.startswith(".") for n in source.parts):
                continue
            body = source.read_text()
            for output_name in self.start_pattern.findall(body):
                if output_name:
                    target = source.parent / output_name
                else:
                    target = source.with_suffix(".png")
                yield (
                    source.relative_to(self.base),
                    target.relative_to(self.base)
                )


class PlantUML:
    conda_env_name = "CaseStudy"
    base_env = Path.home() / "miniconda3" / "envs" / conda_env_name

    def __init__(
            self,
            graphviz: Path = Path("bin") / "dot",
            plantjar: Path = Path("share") / "plantuml.jar",
    ) -> None:
        self.graphviz = self.base_env / graphviz
        self.plantjar = self.base_env / plantjar

    def process(self, source: Path) -> None:
        env = {
            "GRAPHVIZ_DOT": str(self.graphviz),
        }
        command = [
            "java", "-jar", str(self.plantjar), "-progress", str(source)
        ]
        subprocess.run(command, env=env, check=True)
        print()


class GenerateImages:
    def __init__(self, base: Path) -> None:
        self.finder = FindUML(base)
        self.painter = PlantUML()

    def make_all_images(self) -> None:
        for source, target in self.finder.uml_file_iter():
            if (
                    not target.exists()
                    or source.stat().st_mtime > target.stat().st_mtime
            ):
                print(f"Processing {source} -> {target}")
                self.painter.process(source)
            else:
                print(f"Skipping {source} -> {target}")


# if __name__ == "__main__":
#     g = GenerateImages(Path.cwd())
#     g.make_all_images()


class Buffer(Sequence[int]):
    def __init__(self, content: bytes) -> None:
        self.content = content

    def __len__(self) -> int:
        return len(self.content)

    def __iter__(self) -> Iterator[int]:
        return iter(self.content)

    @overload
    def __getitem__(self, index: int) -> int:
        ...

    @overload
    def __getitem__(self, index: slice) -> bytes:
        ...

    @overload
    def __getitem__(self, index: Union[int, slice]) -> Union[int, bytes]:
        return self.content[index]


raw = Buffer(b"$GPGLL,3751.65,S,14507.36,E*77")


class GPSError:
    pass


class Point:
    __slots__ = ("latitude", "longitude")

    def __init__(self, latitude: float, longitude: float) -> None:
        self.latitude = latitude
        self.longitude = longitude

    def __repr__(self) -> str:
        return (
            f"Point(latitude={self.latitude}, "
            f"longitude={self.longitude})"
        )

    @classmethod
    def from_bytes(cls, param, param1, param2, param3):
        pass


class Message(abc.ABC):
    def __init__(self) -> None:
        self.buffer: weakref.ReferenceType[Buffer]
        self.offset: int
        self.end: Optional[int]
        self.commas: list[int]

    def from_buffer(self, buffer: Buffer, offset: int) -> "Message":
        self.buffer = weakref.ref(buffer)
        self.offset = offset
        self.commas = [offset]
        self.end = None
        for index in range(offset, offset + 82):
            if buffer[index] == ord(b","):
                self.commas.append(index)
            elif buffer[index] == ord(b"*"):
                self.commas.append(index)
                self.end = index + 3
                break
        if self.end is None:
            raise GPSError("Incomplete")
        # TODO: confirm checksum
        return self

    def __getitem__(self, field: int) -> bytes:
        if (not hasattr(self, "buffer")
                or (buffer := self.buffer()) is None):
            raise RuntimeError("Broken reference")
        start, end = self.commas[field] + 1, self.commas[field + 1]
        return buffer[start:end]

    def get_fix(self) -> Point:
        return Point.from_bytes(
            self.latitude(),
            self.lat_n_s(),
            self.longitude(),
            self.lon_e_w()
        )

    @abc.abstractmethod
    def latitude(self) -> bytes:
        ...

    @abc.abstractmethod
    def lat_n_s(self) -> bytes:
        ...

    @abc.abstractmethod
    def longitude(self) -> bytes:
        ...

    @abc.abstractmethod
    def lon_e_w(self) -> bytes:
        ...


class GPGLL(Message):
    def latitude(self) -> bytes:
        return self[1]

    def lat_n_s(self) -> bytes:
        return self[2]

    def longitude(self) -> bytes:
        return self[3]

    def lon_e_w(self) -> bytes:
        return self[4]


class GPGGA:
    pass


class GPRMC:
    pass


def message_factory(header: bytes) -> Optional[Message]:
    # TODO: Add functools.lru_cache to save storage and time
    if header == b"GPGGA":
        return GPGGA()
    elif header == b"GPGLL":
        return GPGLL()
    elif header == b"GPRMC":
        return GPRMC()
    else:
        return None


# buffer = Buffer(
#      b"$GPGLL,3751.65,S,14507.36,E*77"
#  )
# flyweight = message_factory(buffer[1:6])
# flyweight.from_buffer(buffer, 0)
# Point(latitude=-37.86083333333, longitude=145.122666666667)
# print(flyweight.get_fix())


# while True:
#     buffer = Buffer(gps_divice.read(1024))
#     # process the message in the buffer


# buffer_2 = Buffer(
#     b"$GPGLL,3751.65,S,14507.36,E*77\\r\\n"
#     b"$GPGLL,3723.2475,N,12158.3416,W,161229.487,A,A*41\\r\\n"
# )
# start = 0
# flyweight = message_factory(buffer_2[start+1:start+6])
# p_1 = flyweight.from_buffer(buffer_2, start).get_fix()
# print(p_1)
#
# print(flyweight.end)
# next_start = buffer_2.index(ord(b"$"), flyweight.end)
# print(next_start)
# flyweight = message_factory(buffer_2[next_start+1:next_start+6])
# p_2 = flyweight.from_buffer(buffer_2, next_start).get_fix()
# print(p_2)


class Suit(str, Enum):
    Clubs = "\N{Black Club Suit}"
    Diamonds = "\N{Black Diamond Suit}"
    Hearts = "\N{Black Heart Suit}"
    Spades = "\N{Black Spade Suit}"


class Card(NamedTuple):
    rank: int
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


class Trick(int, Enum):
    pass


class Hand(List[Card]):
    def __init__(self, *cards: Card) -> None:
        super().__init__(cards)

    def scoring(self) -> List[Trick]:
        pass


class CardGameFactory(abc.ABC):
    @abc.abstractmethod
    def make_card(self, rank: int, suit: Suit) -> "Card":
        ...

    @abc.abstractmethod
    def make_hand(self, *cards: Card) -> "Hand":
        ...


class Cribbage(Card):
    @property
    def points(self) -> int:
        return self.rank


class CribbageAce(Card):
    @property
    def points(self) -> int:
        return 1


class CribbageFace(Card):
    @property
    def points(self) -> int:
        return 10


class CribbageHand(Hand):
    starter: Card

    def upcard(self, starter: Card) -> "Hand":
        self.starter = starter
        return self

    def scoring(self) -> List[Trick]:
        """15's. Pairs. Runs. Right Jack."""
        # ... details imitted ...
        # return tricks


class PokerCard(Card):
    def __str__(self) -> str:
        if self.rank == 14:
            return f"A{self.suit}"
        return f"{self.rank}{self.suit}"


class PokerHand(Hand):
    def scoring(self) -> List[Trick]:
        """Return a single 'Trick"""
        # ... details omitted ...
        # return [rank]


class PokerFactory(CardGameFactory):
    def make_card(self, rank: int, suit: Suit) -> "Card":
        if rank == 1:
            # Ace above kings
            rank = 14
        return PokerCard(rank, suit)

    def make_hand(self, *cards: Card) -> "Hand":
        return PokerHand(*cards)


# factory = PokerFactory()
# cards = [
#     factory.make_card(6, Suit.Clubs),
#     factory.make_card(7, Suit.Diamonds),
#     factory.make_card(8, Suit.Hearts),
#     factory.make_card(9, Suit.Spades),
# ]
# starter = factory.make_card(5, Suit.Spades)
# hand = factory.make_hand(*cards)
# score = sorted(hand.upcard(starter).scoring())
# [t.name for t in score]


class CardGameFactoryProtocol(Protocol):
    def make_card(self, rank: int, suit: Suit) -> "Card":
        ...

    def make_hand(self, *cards: Card) -> "Hand":
        ...


class Folder:
    def __init__(
            self,
            name: str,
            children: Optional[dict[str, "Node"]] = None
    ) -> None:
        self.name = name
        self.children = children or {}
        self.parent: Optional["Folder"] = None

    def __repr__(self) -> str:
        return f"Folder({self.name!r}, {self.children!r})"

    def add_child(self, node: "Node") -> "Node":
        node.parent = self
        return self.children.setdefault(node.name, node)

    def move(self, new_folder: "Folder") -> None:
        pass

    def copy(self, new_folder: "Folder") -> None:
        pass

    def remove(self) -> None:
        pass


class File:
    def __init__(self, name: str) -> None:
        self.name = name
        self.parent: Optional[Folder] = None

    def __repr__(self) -> str:
        return f"File({self.name!r})"

    def move(self, new_path):
        pass

    def copy(self, new_path):
        pass

    def remove(self):
        pass


class Node(abc.ABC):
    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name
        self.parent: Optional["Folder"] = None

    def move(self, new_place: "Folder") -> None:
        previous = self.parent
        new_place.add_child(self)
        if previous:
            del previous.children[self.name]

    @abc.abstractmethod
    def copy(self, new_folder: "Folder") -> None:
        ...

    @abc.abstractmethod
    def remove(self) -> None:
        ...


class Folderbis(Node):
    def __init__(
            self,
            name: str,
            children: Optional[dict[str, "Node"]] = None
    ) -> None:
        super().__init__(name)
        self.children = children or {}

    def __repr__(self) -> str:
        return f"Folder({self.name!r}, {self.children!r})"

    def add_child(self, node: "Node") -> "Node":
        node.parent = self
        return self.children.setdefault(node.name, node)

    def copy(self, new_folder: "Folder") -> None:
        target = new_folder.add_child(Folderbis(self.name))
        for c in self.children:
            self.children[c].copy(target)

    def remove(self) -> None:
        names = list(self.children)
        for c in names:
            self.children[c].remove()
        if self.parent:
            del self.parent.children[self.name]


class Filebis(Node):
    def __repr__(self) -> str:
        return f"File({self.name!r})"

    def copy(self, new_folder: "Folderbis") -> None:
        new_folder.add_child(Filebis(self.name))

    def remove(self) -> None:
        if self.parent:
            del self.parent.children[self.name]


tree = Folderbis("Tree")
tree.add_child(Folderbis("src"))
tree.children["src"].add_child(Filebis("ex1.py"))
tree.add_child(Folderbis("src"))
tree.children["src"].add_child(Filebis("test1.py"))
# print(tree)

test1 = tree.children["src"].children["test1.py"]
# print(test1)
tree.add_child(Folderbis("tests"))
test1.move(tree.children["tests"])
# print(tree)


def test_setup(db_name: str = "sales.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_name)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Sales (
            salesperson text,
            amt currency,
            year integer,
            model text,
            new boolean
        )
        """
    )
    conn.execute(
        """
        DELETE FROM Sales
        """
    )
    conn.execute(
        """
        INSERT INTO Sales
        VALUES('Tim', 16000, 2010, 'Honda Fit', 'true')
        """
    )
    conn.execute(
        """
        INSERT INTO Sales
        VALUES('Tim' 9000, 2006, 'Ford Focus', 'false')
        """
    )
    conn.execute(
        """
        INSERT INTO Sales
        VALUES('Hannah', 8000, 2004, 'Dodge Neon', 'false')
        """
    )
    conn.execute(
        """
        INSERT INTO Sales
        VALUES('Hannah', 28000, 2009, 'Ford Mustang', 'true')
        """
    )
    conn.execute(
        """
        INSERT INTO Sales
        VALUES('Hannah', 50000, 2010, 'Lincoln Navigator', 'true')
        """
    )
    conn.execute(
        """
        INSERT INTO Sales
        VALUES('Jason', 20000, 2008, 'Toyota Prius', 'false')
        """
    )
    conn.commit()
    return conn


class QueryTemplate:
    def __init__(self, db_name: str = "sales.db") -> None:
        self.db_name = db_name
        self.conn: sqlite3.Connection
        self.results: list[tuple[str, ...]]
        self.query: str
        self.header: list[str]

    def connect(self) -> None:
        self.conn = sqlite3.connect(self.db_name)

    def construct_query(self) -> None:
        raise NotImplementedError("construct_query not implemented")

    def do_query(self) -> None:
        results = self.conn.execute(self.query)
        self.results = results.fetchall()

    def output_context(self) -> ContextManager[TextIO]:
        self.target_file = sys.stdout
        return cast(ContextManager[TextIO], contextlib.nullcontext())

    def output_results(self) -> None:
        writer = csv.writer(self.target_file)
        writer.writerow(self.header)
        writer.writerows(self.results)

    def process_format(self) -> None:
        self.connect()
        self.construct_query()
        self.do_query()
        with self.output_context():
            self.output_results()


class NewVehiclesQuery(QueryTemplate):
    def construct_query(self) -> None:
        self.query = "select * from Sales where new='true'"
        self.header = ["salesperson", "amt", "year", "model", "new"]


class SalesGrossQuery(QueryTemplate):
    def construct_query(self) -> None:
        self.query = (
            "select salesperson, sum(amt) "
            " from Sales group by salesperson"
        )
        self.header = ["salesperson", "total sales"]

    def output_context(self) -> ContextManager[TextIO]:
        today = datetime.date.today()
        filepath = Path(f"gross_sales_{today:%Y%m%d}.csv")
        self.target_file = filepath.open("w")
        return self.target_file








