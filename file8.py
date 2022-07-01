from __future__ import annotations

import ast
import csv
import logging
import logging.handlers
import weakref
import asyncio
import asyncio.exceptions
import httpx
from concurrent import futures
from contextlib import contextmanager
from fnmatch import fnmatch
from functools import wraps, lru_cache
from pprint import pprint
from queue import Queue
from threading import Timer, Thread, Lock
from multiprocessing import Process, cpu_count
from multiprocessing.pool import Pool
from urllib.request import urlopen
import datetime
import math
import random
import pickle
import json
import signal
import struct
import subprocess
from math import radians, pi, hypot, cos, factorial, sqrt, ceil
from pathlib import Path
import abc
from PIL import Image
from types import TracebackType
from decimal import Decimal
from typing import NamedTuple, Protocol, Sequence, Tuple, Match, Any, Iterable, Iterator, Optional, List, Dict, TextIO, \
    Callable, cast, Type, Literal, Pattern, Match, overload, Union, ContextManager, TYPE_CHECKING, Set
from urllib.parse import urlparse
from urllib.request import urlopen
from enum import Enum, auto
from itertools import permutations
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

THE_ORDERS = [
    "Reuben",
    "Ham and Cheese",
    "Monte Cristo",
    "Tuna Melt",
    "Cuban",
    "Grilled Cheese",
    "French Dip",
    "BLT",
]


class Chef(Thread):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.total = 0

    def get_order(self) -> None:
        self.order = THE_ORDERS.pop(0)

    def prepare(self) -> None:
        """Simulate doing a lot of work with a BIG computation"""
        start = time.monotonic()
        target = start + 1 + random.random()
        for i in range(1_000_000_000):
            self.total += math.factorial(i)
            if time.monotonic() >= target:
                break
        print(
            f"{time.monotonic():.3f} {self.name} made {self.order}"
        )

    def run(self) -> None:
        while True:
            try:
                self.get_order()
                self.prepare()
            except IndexError:
                break  # No more orders


# Mo = Chef("Michael")
# Constantine = Chef("Constantine")
# if __name__ == "__main__":
#     random.seed(42)
#     Mo.start()
#     Constantine.start()


class MuchCPU(Process):
    def run(self) -> None:
        print(f"OS PID {os.getpid()}")
        s = sum(
            2 * i + 1 for i in range(100_000_000)
        )


# if __name__ == "__main__":
#     workers = [MuchCPU() for f in range(cpu_count())]
#     t = time.perf_counter()
#     for p in workers:
#         p.start()
#     for p in workers:
#         p.join()
#     print(f"work took {time.perf_counter() - t:.3f} seconds")


def prime_factors(value: int) -> list[int]:
    if value in {2, 3}:
        return [value]
    factors: list[int] = []
    for divisor in range(2, ceil(sqrt(value)) + 1):
        quotient, remainder = divmod(value, divisor)
        if not remainder:
            factors.extend(prime_factors(divisor))
            factors.extend(prime_factors(quotient))
            break
        else:
            factors = [value]
        return factors


# if __name__ == "__main__":
#     to_factor = [
#         random.randint(100_000_000, 1_000_000_000)
#         for i in range(40_960)
#     ]
#     with Pool() as pool:
#         results = pool.map(prime_factors, to_factor)
#         primes = [
#             value
#             for value, factor_list in zip(to_factor, results)
#             if len(factor_list) == 1
#         ]
#         print(f"9-digit primes {primes}")


if TYPE_CHECKING:
    Query_Q = Queue[Union[str, None]]
    Result_Q = Queue[List[str]]


def search(
        paths: List[Path],
        query_q: Query_Q,
        results_q: Result_Q
) -> None:
    print(f"PID: {os.getpid()}, paths {len(paths)}")
    lines: List[str] = []
    for path in paths:
        lines.extend(
            l.rstrip() for l in path.read_text().splitlines()
        )
    while True:
        if (query_text := query_q.get()) is None:
            break
        results = [l for l in lines if query_text in l]
        results_q.put(results)


class DirectorySearch:
    def __init__(self) -> None:
        self.query_queues: List[Query_Q]
        self.results_queue: Result_Q
        self.search_workers: List[Process]

    def setup_search(
            self, paths: List[Path], cpus: Optional[int] = None
    ) -> None:
        if cpus is None:
            cpus = cpu_count()
        worker_paths = [paths[i::cpus] for i in range(cpus)]
        self.query_queues = [Queue() for p in range(cpus)]
        self.results_queue = Queue()
        self.search_workers = [
            Process(
                target=search, args=(paths, q, self.results_queue))
            for paths, q in zip(worker_paths, self.query_queues)
        ]
        for proc in self.search_workers:
            proc.start()

    def teardown_search(self) -> None:
        # signal process termination
        for q in self.query_queues:
            q.put(None)
        for proc in self.search_workers:
            proc.join()

    def search(self, target: str) -> Iterator[str]:
        for q in self.query_queues:
            q.put(target)
        for i in range(len(self.results_queue)):
            for match in self.results_queue.get():
                yield match


def all_source(path: Path, pattern: str) -> Iterator[Path]:
    for root, dirs, files in os.walk(path):
        for skip in {".tox", ".mypy_cache", "__pycache__", ".idea"}:
            if skip in dirs:
                dirs.remove(skip)
        yield from (
            Path(root) / f for f in files if fnmatch(f, pattern)
        )


# if __name__ == "__main__":
#     ds = DirectorySearch()
#     base = Path.cwd().parent
#     all_paths = list(all_source(base, "*.py"))
#     ds.setup_search(all_paths)
#     for target in ("import", "class", "def"):
#         start = time.perf_counter()
#         count = 0
#         for line in ds.search(target):
#             # print(line)
#             count += 1
#         milliseconds = 1000*(time.perf_counter() - start)
#         print(
#             f"Found {count} {target!r} in {len(all_paths)} files "
#             f"in {milliseconds:.3f}ms"
#         )
#     ds.teardown_search()


class ImportResult(NamedTuple):
    path: Path
    imports: Set[str]

    @property
    def focus(self) -> bool:
        return "typing" in self.imports


class ImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: Set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module)


def find_imports(path: Path) -> ImportResult:
    tree = ast.parse(path.read_text())
    iv = ImportVisitor()
    iv.visit(tree)
    return ImportResult(path, iv.imports)


def main() -> None:
    start = time.perf_counter()
    base = Path.cwd().parent
    with futures.ThreadPoolExecutor(24) as pool:
        analyzers = [
            pool.submit(find_imports, path)
            for path in all_source(base, "*.py")
        ]
        analyzed = (
            worker.result()
            for worker in futures.as_completed(analyzers)
        )
    for example in sorted(analyzed):
        print(
            f"{'->' if example.focus else '':2s} "
            f"{example.path.relative_to(base)} {example.imports}"
        )
    end = time.perf_counter()
    rate = 1000 * (end - start) / len(analyzers)
    print(f"Searched {len(analyzers)} files at {rate:.3f}ms/file")


async def random_sleep(counter: float) -> None:
    delay = random.random() * 5
    print(f"{counter} sleeps for {delay:.2f} seconds")
    await asyncio.sleep(delay)
    print(f"{counter} awakens, refreshed")


async def sleepers(how_many: int = 5) -> None:
    print(f"Creating {how_many} tasks")
    tasks = [
        asyncio.create_task(random_sleep(i))
        for i in range(how_many)
    ]
    print(f"Waiting for {how_many} tasks")
    await asyncio.gather(*tasks)


# if __name__ == "__main__":
#     asyncio.run(sleepers(5))
#     print("Done with the sleepers")


SIZE_FORMAT = ">L"
SIZE_BYTES = struct.calcsize(SIZE_FORMAT)

TARGET: TextIO
LINE_COUNT = 0


def serialize(bytes_payload: bytes) -> str:
    object_payload = pickle.loads(bytes_payload)
    text_message = json.dumps(object_payload)
    TARGET.write(text_message)
    TARGET.write("\n")
    return text_message


async def log_writer(bytes_payload: bytes) -> None:
    global LINE_COUNT
    LINE_COUNT += 1
    text_message = await asyncio.to_thread(serialize, bytes_payload)


async def log_catcher(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    count = 0
    client_socket = writer.get_extra_info("socket")
    size_header = await reader.read(SIZE_BYTES)
    while size_header:
        payload_size = struct.unpack(SIZE_FORMAT, size_header)
        bytes_payload = await reader.read(payload_size[0])
        await log_writer(bytes_payload)
        count += 1
        size_header = await reader.read(SIZE_BYTES)
    print(f"From {client_socket.getpeername()}: {count} lines")


server: asyncio.AbstractServer


async def mainbis(host: str, port: int) -> None:
    global server
    server = await asyncio.start_server(
        log_catcher,
        host=host,
        port=port,
    )
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, server.close)
    if server.sockets:
        addr = server.sockets[0].getsockname()
        print(f"Serving on {addr}")
    else:
        raise ValueError("Failed to create server")
    async with server:
        await server.serve_forever()


if sys.platform == "win32":
    from types import FrameType


    def close_server(signum: int, frame: FrameType) -> None:
        print(f"Signal {signum}")
        server.close()


    signal.signal(signal.SIGINT, close_server)
    signal.signal(signal.SIGTERM, close_server)
    signal.signal(signal.SIGABRT, close_server)
    signal.signal(signal.SIGBREAK, close_server)

# if __name__ == "__main__":
#     # these often have command-line or environment overrides
#     HOST, PORT = "localhost", 18842
#     with Path("one.log").open("w") as TARGET:
#         try:
#             if sys.platform == "win32":
#                 # https://github.com/encode/httpx/issues/914
#                 loop = asyncio.get_event_loop()
#                 loop.run_until_complete(mainbis(HOST, PORT))
#                 loop.run_until_complete(asyncio.sleep(1))
#                 loop.close()
#             else:
#                 asyncio.run(mainbis(HOST, PORT))
#         except (
#             asyncio.exceptions.CancelledError,
#             KeyboardInterrupt
#         ):
#             ending = {"lines_collected": LINE_COUNT}
#             print(ending)
#             TARGET.write(json.dumps(ending) + "\n")


logger = logging.getLogger(f"app_{os.getpid()}")


class Sorter(abc.ABC):
    def __init__(self) -> None:
        id = os.getpid()
        self.logger = logging.getLogger(f"app_{id}.{self.__class__.__name__}")

    @abc.abstractmethod
    def sort(self, data: list[float]) -> list[float]:
        ...


class BogoSort(Sorter):
    @staticmethod
    def is_ordered(data: tuple[float, ...]) -> bool:
        pairs: Iterable[Tuple[float, float]] = zip(data, data[1:])
        return all(a <= b for a, b in pairs)

    def sort(self, data: list[float]) -> list[float]:
        self.logger.info("Sorting %d", len(data))
        start = time.perf_counter()
        ordering: Tuple[float, ...] = tuple(data[:])
        permute_iter = permutations(data)
        steps = 0
        while not BogoSort.is_ordered(ordering):
            ordering = next(permute_iter)
            steps += 1
            duration = 1000 * (time.perf_counter() - start)
            self.logger.info(
                "Sorted %d items in %d steps, %.3f ms",
                len(data), steps, duration
            )
        return list(ordering)


def main1(workload: int, sorter: Sorter = BogoSort()) -> int:
    total = 0
    for i in range(workload):
        samples = random.randint(3, 10)
        data = [random.random() for _ in range(samples)]
        ordered = sorter.sort(data)
        total += samples
    return total


# if __name__ == "__main__":
#     LOG_HOST, LOG_PORT = "localhost", 18842
#     socket_handler = logging.handlers.SocketHandler(
#         LOG_HOST, LOG_PORT
#     )
#     stream_handler = logging.StreamHandler(sys.stderr)
#     logging.basicConfig(
#         handlers=[socket_handler, stream_handler],
#         level=logging.INFO
#     )
#     start = time.perf_counter()
#     workload = random.randint(10, 20)
#     logger.info("sorting %d collections", workload)
#     samples = main1(workload, BogoSort())
#     end = time.perf_counter()
#     logger.info(
#         "sorted %d collections, taking %f s", workload, end - start
#     )
#     logging.shutdown()


class Zone(NamedTuple):
    zone_name: str
    zone_code: struct
    same_code: str  # Special Area Messaging Encoder

    @property
    def forecast_url(self) -> str:
        return (
            f"https://tgftp.nws.noaa.gov/data/forecasts"
            f"/marine/coastal/an/{self.zone_code.lower()}.txt"
        )


ZONES = [
    Zone("Chesapeake Bay from Pooles Island to Sanday Point, MD",
         "ANZ531", "073531"),
    Zone("Chesapeake Bay from Sandy Point to North Beach, MD",
         "ANZ532", "073532"),
    ...
]


class MarineWX:
    advisory_pat = re.compile(r"\n\.\.\.(.*?)\.\.\.\n", re.M | re.S)

    def __init__(self, zone: Zone) -> None:
        super().__init__()
        self.zone  =zone
        self.doc = ""

    async def run(self) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(self.zone.forecast_url)
        self.doc = response.text

    @property
    def advisory(self) -> str:
        if (match := self.advisory_pat.search(self.doc)):
            return match.group(1).replace("\n", " ")
        return ""

    def __repr__(self) -> str:
        return f"{self.zone.zone_name} {self.advisory}"


async def task_main() -> None:
    start = time.perf_counter()
    forecasts = [MarineWX(z) for z in ZONES]
    await asyncio.gather(
        *(asyncio.create_task(f.run()) for f in forecasts)
    )
    for f in forecasts:
        print(f)
    print(
        f"Got {len(forecasts)} forecasts "
        f"in {time.perf_counter() - start:.3f} seconds"
    )


# if __name__ == "__task_main__":
#     asyncio.run(task_main())


FORKS: List[asyncio.Lock]


async def philosopher(
        id: int,
        footman: asyncio.Semaphore
) -> tuple[int, float, float]:
    async with footman:
        async with FORKS[id], FORKS[(id + 1) % len(FORKS)]:
            eat_time = 1 + random.random()
            print(f"{id} eating")
            await asyncio.sleep(eat_time)
        think_time = 1 + random.random()
        print(f"{id} philosophizing")
        await asyncio.sleep(think_time)
    return id, eat_time, think_time


async def mainn(faculty: int = 5, servings: int = 5) -> None:
    global FORKS
    FORKS = [asyncio.Lock() for i in range(faculty)]
    footman = asyncio.BoundedSemaphore(faculty - 1)
    for serving in range(servings):
        departement = (
            philosopher(p, footman) for p in range(faculty)
        )
        results = await asyncio.gather(*departement)
        print(results)


# if __name__ == "__main__":
#     asyncio.run(mainn())




