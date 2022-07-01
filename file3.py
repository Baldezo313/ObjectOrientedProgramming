from __future__ import annotations
import string
import operator
import random
import fnmatch
import re
import zipfile
import datetime
import time
import logging
import collections
from collections.abc import Container
from collections import defaultdict, Counter
from pathlib import Path
import abc
from abc import ABC, abstractmethod
from dataclasses import dataclass
# from PIL import Image
from decimal import Decimal
from math import hypot
from urllib.request import urlopen

# from mypy.typeshed.stdlib.tkinter import Image
from database import Database
from typing import List, Optional, Protocol, Any, NoReturn, Union, Tuple, Iterable, cast, Type, Set, Dict, Hashable, Mapping, TypedDict, NamedTuple
from typing import Deque, TYPE_CHECKING
import queue
from functools import wraps, total_ordering
from pprint import pprint


class WebPage:
    def __init__(self, url: str) -> None:
        self.url = url
        self._content: Optional[bytes] = None

    @property
    def content(self) -> bytes:
        if self._content is None:
            print("Retrieving New Page...")
            with urlopen(self.url) as response:
                self._content = response.read()
        return self._content


webpage = WebPage("http://ccphillips.net/")
now = time.perf_counter()
content1 = webpage.content
first_fetch = time.perf_counter() - now
now = time.perf_counter()
content2 = webpage.content
second_fetch = time.perf_counter() - now
assert content2 == content1, "Problem: Pages were different"

print(f"Initial Request     {first_fetch:.5f}")
print(f"Subsequent Requests {second_fetch:.5f}")


class AverageList(List[int]):
    @property
    def average(self) -> float:
        return sum(self) / len(self)


a = AverageList([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
print(a.average)


class ZipReplace:
    def __init__(
            self,
            archive: Path,
            pattern: str,
            find: str,
            replace: str
    ) -> None:
        self.archive_path = archive
        self.pattern = pattern
        self.find = find
        self.replace = replace

    def find_and_replace(self) -> None:
        input_path, output_path = self.make_backup()
        with zipfile.ZipFile(output_path, "w") as output:
            with zipfile.ZipFile(input_path) as input:
                self.copy_and_transform(input, output)

    def make_backup(self) -> tuple[Path, Path]:
        input_path = self.archive_path.with_suffix(
            f"{self.archive_path.suffix}.old"
        )
        output_path = self.archive_path
        self.archive_path.rename(input_path)
        return input_path, output_path

    def copy_and_transform(self, input: zipfile.ZipFile, output: zipfile.ZipFile) -> None:
        for item in input.infolist():
            extracted = Path(input.extract(item))
            if not item.is_dir() and fnmatch.fnmatch(item.filename, self.pattern):
                print(f"Transform {item}")
                input_text = extracted.read_text()
                output_text = re.sub(self.find, self.replace, input_text)
                extracted.write_text(output_text)
            else:
                print(f"Ignore {item}")
            output.write(extracted, item.filename)
            extracted.unlink()
            for parent in extracted.parents:
                if parent == Path.cwd():
                    break
                parent.rmdir()


sample = Sample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa")
if __name__ == "__main__":
    sample_zip = Path("C:/Users/balde/Desktop/Oreilli/OOP1/archive (1).zip")
    zr = ZipReplace(sample_zip, "*.md", "xyzzy", "plover's egg")
    zr.find_and_replace()


class ZipProcessor(ABC):
    def __init__(self, archive: Path) -> None:
        self.archive_path = archive
        self._pattern :str

    def process_files(self, pattern: str) -> None:
        self._pattern = pattern
        input_path, output_path = self.make_backup()
        with zipfile.ZipFile(output_path, "w") as output:
            with zipfile.ZipFile(input_path) as input:
                self.copy_and_transform(input, output)

    def make_backup(self) -> tuple[Path, Path]:
        input_path = self.archive_path.with_suffix(
            f"{self.archive_path.suffix}.old"
        )
        output_path = self.archive_path
        self.archive_path.rename(input_path)
        return input_path, output_path

    def copy_and_transform(
        self, input: zipfile.ZipFile, output: zipfile.ZipFile
    ) -> None:
        for item in input.infolist():
            extracted = Path(input.extract(item))
            if self.matches(item):
                print(f"Transform {item}")
                self.transform(extracted)
            else:
                print(f"Ignore {item}")
            output.write(extracted, item.filename)
            self.remove_under_cwd(extracted)

    def matches(self, item: zipfile.ZipInfo) -> bool:
        return (
            not item.is_dir()
            and fnmatch.fnmatch(item.filename, self._pattern)
        )

    def remove_under_cwd(self, extracted: Path) -> None:
        extracted.unlink()
        for parent in extracted.parents:
            if parent == Path.cwd():
                break
            parent.rmdir()

    @abstractmethod
    def transform(self, extracted: Path) -> None:
        pass


class TextTweaker(ZipProcessor):
    def __init__(self, archive: Path) -> None:
        super().__init__(archive)
        self.find: str
        self.replace: str

    def find_and_replace(self, find: str, replace: str) -> "TextTweaker":
        self.find = find
        self.replace = replace
        return self

    def transform(self, extracted: Path) -> None:
        input_text = extracted.read_text()
        output_text = re.sub(self.find, self.replace, input_text)
        extracted.write_text(output_text)


class ImgTweaker(ZipProcessor):
    def transform(self, extracted: Path) -> None:
        image = Image.open(extracted)
        scaled = image.resize(size=(640, 960))
        scaled.save(extracted)


class OddIntegers:
    def __contains__(self, x: int) -> bool:
        return x % 2 != 0


odd = OddIntegers()
# print(isinstance(odd, Container))
# print(issubclass(OddIntegers,  Container))
print(1 in odd)
print(2 in odd)
print(3 in odd)


class Die(abc.ABC):
    def __init__(self) -> None:
        self.face: int
        self.roll()

    @abc.abstractmethod
    def roll(self) -> None:
        ...

    def __repr__(self) -> str:
        return f"{self.face}"


class Bad(Die):
    def roll(self, a:int, b:int) -> float:
        return (a+b)/2


class D4(Die):
    def roll(self) -> None:
        self.face = random.choice((1, 2, 3, 4))


class D6(Die):
    def roll(self) -> None:
        self.face = random.randint(1, 6)


class Dice(abc.ABC):
    def __init__(self, n: int, die_class: Type[Die]) -> None:
        self.dice = [die_class() for _ in range(n)]

    @abc.abstractmethod
    def roll(self) -> None:
        ...

    @property
    def total(self) -> int:
        return sum(d.face for d in self.dice)


class SimpleDice(Dice):
    def roll(self) -> None:
        for d in self.dice:
            d.roll()


# sd = SimpleDice(6, D6)
# print(sd.roll())
# print(sd.total)


class YachtDice(Dice):
    def __init__(self) -> None:
        super().__init__(5, D6)
        self.saved: Set[int] = set()

    def saving(self, positions: Iterable[int]) -> "YachtDice":
        if not all(0 <= n < 6 for n in positions):
            raise ValueError("Invalid position")
        self.saved = set(positions)
        return self

    def roll(self) -> None:
        for n, d in enumerate(self.dice):
            if n not in self.saved:
                d.roll()
        self.saved = set()


sd = YachtDice()
sd.roll()
print(sd.dice)
sd.saving([0, 1, 2]).roll()
print(sd.dice)

home = Path.home()
print(home)


class DDice:
    def __init__(self, *die_class: Type[Die]) -> None:
        self.dice = [dc() for dc in die_class]
        self.adjust: int = 0

    def plus(self, adjust: int = 0) -> "DDice":
        self.adjust = adjust
        return self

    def roll(self) -> None:
        for d in self.dice:
            d.roll()

    @property
    def total(self) -> int:
        return sum(d.face for d in self.dice) + self.adjust

    def __add__(self, die_class: Any) -> "DDice":
        if isinstance(die_class, type) and issubclass(die_class, Die):
            new_classes = [type(d) for d in self.dice] + [die_class]
            new = DDice(*new_classes).plus(self.adjust)
            return new
        elif isinstance(die_class, int):
            new_classes = [type(d) for d in self.dice]
            new = DDice(*new_classes).plus(die_class)
            return new
        else:
            return NotImplemented

    def __radd__(self, die_class: Any) -> "DDice":
        if isinstance(die_class, type) and issubclass(die_class, Die):
            new_classes = [die_class] + [type(d) for d in self.dice]
            new = DDice(*new_classes).plus(self.adjust)
            return new
        elif isinstance(die_class, int):
            new_classes = [type(d) for d in self.dice]
            new = DDice(*new_classes).plus(die_class)
            return new
        else:
            return NotImplemented

    def __mul__(self, n: Any) -> "DDice":
        if isinstance(n, int):
            new_classes = [type(d) for d in self.dice for _ in range(n)]
            return DDice(*new_classes).plus(self.adjust)
        else:
            return NotImplemented

    def __rmul__(self, n: Any) -> "DDice":
        if isinstance(n, int):
            new_classes = [type(d) for d in self.dice for _ in range(n)]
            return DDice(*new_classes).plus(self.adjust)
        else:
            return NotImplemented

    def __iadd__(self, die_class: Any) -> "DDice":
        if isinstance(die_class, type) and issubclass(die_class, Die):
            self.dice += [die_class()]
            return self
        elif isinstance(die_class, int):
            self.adjust += die_class
            return self
        else:
            return NotImplemented


class NoDupDict(Dict[Hashable, Any]):
    def __setitem__(self, key, value) -> None:
        if key in self:
            raise ValueError(f"duplicate {key!r}")
        super().__setitem__(key, value)


nd = NoDupDict()
nd["a"] = 1
nd["a"] = 2

DictInit = Union[Iterable[Tuple[Hashable, Any]], Mapping[Hashable, Any], None]


class NoDupDict(Dict[Hashable, Any]):
    def __setitem__(self, key: Hashable, value: Any) -> None:
        if key in self:
            raise ValueError(f"duplicate {key!r}")
        super().__setitem__(key, value)

    def __init__(self, init: DictInit = None, **kwargs: Any) -> None:
        if isinstance(init, Mapping):
            super().__init__(init, **kwargs)
        elif isinstance(init, Iterable):
            for k, v in cast(Iterable[Tuple[Hashable, Any]], init):
                self[k] = v
        elif init is None:
            super().__init__(**kwargs)
        else:
            super().__init__(init, **kwargs)


class DieMeta(abc.ABCMeta):
    def __new__(
        metaclass: Type[type],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> "DieMeta":
        if "roll" in namespace and not getattr(
            namespace["roll"], "__isabstractmethod__", False
        ):
            namespace.setdefault("logger", logging.getLogger(name))
            original_method = namespace["roll"]

            @wraps(original_method)
            def logged_roll(self: "DieLog") -> None:
                original_method(self)
                self.logger.info(f"Rolled {self.face}")

            namespace["roll"] = logged_roll()
        new_object = cast(
            "DieMeta", abc.ABCMeta.__new__(
                metaclass, name, bases, namespace
            )
        )
        return new_object



class MyObject:
    pass


m = MyObject()
m.x = "hello"
print(m.x)


def middle(stock, date):
    symbol, current, high, low = stock
    return (((high + low) / 2), date)

# print(middle(("AAPL", 123.52, 53.15, 137.98), datetime.date(2020, 12, 4)))


s = "AAPL", 123.76, 134.80, 130.53
high = s[2]
kk = s[1:3]
print(high)
print(kk)


def high(stock):
    symbol, current, high, low = stock
    return high

print(high(s))


class Stock(NamedTuple):
    symbol: str
    current: float
    high: float
    low: float


# print(Stock("AAPL", 123.52, 137.98, 53.15))
s2 = Stock(symbol='AAPL', current=123.52, high=137.98, low=53.15)
print(s2.high)
print(s2[2])
symbol, current, high, low = s2
print(current)

t = ("Relayer", ["Gates of Delirium", "Sound Chaser"])
t[1].append("To Be Over")
print(t)


class Stock(NamedTuple):
    symbol: str
    current: float
    high: float
    low: float

    @property
    def middle(self) -> float:
        return (self.high + self.low)/2


s = Stock("AAPL", 123.52, 137.98, 53.15)
print(s.middle)


@dataclass
class Stock:
    symbol: str
    current: float
    high: float
    low: float


s = Stock("AAPL", 123.52, 137.98, 53.15)
# # print(s)
# # print(s.current)
s.current = 122.25
# # print(s)
# s.unexpected_attribute = 'allowed'
# print(s.unexpected_attribute)


class StockOrdinary:
    def __init__(self, name: str, current: float, high: float, low: float) -> None:
        self.name = name
        self.current = current
        self.high = high
        self.low = low


s_ord = StockOrdinary("AAPL", 123.52, 137.98, 53.15)
# print(s_ord)
s_ord_2 = StockOrdinary("AAPL", 123.52, 137.98, 53.15)
# print(s_ord == s_ord_2)
stock2 = Stock(symbol='AAPL', current=122.25, high=137.98, low=53.15)
# print(s == stock2)


@dataclass
class StockDefaults:
    name: str
    current: float = 0.0
    high: float = 0.0
    low: float = 0.0


# print(StockDefaults("GOOG"))
# print(StockDefaults("GOOG", 1826.77, 1847.20, 1013.54))


@dataclass(order=True)
class StockOrder:
    name: str
    current: float = 0.0
    high: float = 0.0
    low: float = 0.0


stock_ordered1 = StockOrder("GOOG", 1826.77, 1847.20, 1013.54)
stock_ordered2 = StockOrder("GOOG")
stock_ordered3 = StockOrder("GOOG", 1728.28, high=1733.18, low=1666.33)
print(stock_ordered1 < stock_ordered2)
print(stock_ordered1 > stock_ordered2)

pprint(sorted([stock_ordered1, stock_ordered2, stock_ordered3]))


stocks = {"GOOG": (1235.20, 1242.54, 1231.06), "MSFT": (110.41, 110.45, 109.84)}
print(stocks["GOOG"])
print(stocks.get("RIMM"))
print(stocks.get("RIMM", "NOT FOUND"))
print(stocks.setdefault("GOOG", "INVALID"))
print(stocks.setdefault("BB", (10.87, 10.76, 10.90)))
print(stocks["BB"])

for stock, values in stocks.items():
    print(f"{stock} last value is {values[0]}")

stocks["GOOG"] = (1245.21, 1252.64, 1245.18)
print(stocks['GOOG'])

random_keys = {}
random_keys["astring"] = "somestring"
random_keys[5] = "aninteger"
random_keys[25.2] = "float work too"
random_keys[("abc", 123)] = "so do tuples"


class AnObject:
    def __init__(self, avalue):
        self.avalue = avalue


my_object = AnObject(14)
random_keys[my_object] = "we can even store objects"
my_object.avalue = 12
# random_keys[[1, 2, 3]] = "we can't use lists as keys"

for key in random_keys:
    print(f"{key!r} has value {random_keys[key]!r}")


def letter_frequency(sentence: str) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for letter in sentence:
        frequency = frequencies.setdefault(letter, 0)
        frequencies[letter] = frequency + 1
    return frequencies


def letter_frequency_2(sentence: str) -> defaultdict[str, int]:
    frequencies: defaultdict[str, int] = defaultdict(int)
    for letter in sentence:
        frequencies[letter] += 1
    return frequencies


@dataclass
class Prices:
    current: float = 0.0
    high: float = 0.0
    low: float = 0.0


# print(Prices())

portfolio = collections.defaultdict(Prices)
print(portfolio["GOOG"])

portfolio["AAPL"] = Prices(current=122.25, high=137.98, low=53.15)
pprint(portfolio)


def make_defauldict():
    return collections.defaultdict(Prices)


by_month = collections.defaultdict(lambda : collections.defaultdict(Prices))
by_month["AAPL"]["Jan"] = Prices(current=122.25, high=137.98, low=53.15)


def leeter_frequency_3(sentence: str) -> Counter[str]:
    return Counter(sentence)


responses = ["vanilla", "chocolate", "vanilla", "vanilla", "vanilla", "caramel", "strawberry", "vanilla"]
favorites = collections.Counter(responses).most_common(1)
name, frequency = favorites[0]
print(name)


CHARACTERS = list(string.ascii_letters) + [" "]


def letter_frequency(sentence: str) -> list[tuple[str, int]]:
    frequencies = [(c, 0) for c in CHARACTERS]
    for letter in sentence:
        index = CHARACTERS.index(letter)
        frequencies[index] = (letter, frequencies[index][1] + 1)
    non_zero = [
        (letter, count)
        for letter, count in frequencies if count > 0
    ]
    return non_zero


@dataclass(frozen=True)
class MultiItem:
    data_source: str
    timestamp: Optional[float]
    creation_date: Optional[str]
    name: str
    owner_etc: str

    def __lt__(self, other: Any) -> bool:
        if self.data_source == "Local":
            self_datetime = datetime.datetime.fromtimestamp(
                cast(float, self.timestamp)
            )
        else:
            self_datetime = datetime.datetime.fromisoformat(
                cast(str, self.creation_date)
            )
        if other.data_source == "Local":
            other_datetime = datetime.datetime.fromtimestamp(
                cast(float, other.timestamp)
            )
        else:
            other_datetime = datetime.datetime.fromisoformat(
                cast(str, other.creation_date)
            )
        return self_datetime < other_datetime


mi_0 = MultiItem("Local", 1607280522.68012, None, "Some File", "etc. 0")
mi_1 = MultiItem("Remote", None, "2020-12-06T13:47:52.849153", "Another File", "etc. 1")
mi_2 = MultiItem("Local", 1579373292.452993, None, "This File", "etc. 2")
mi_3 = MultiItem("Remote", None, "2020-01-18T13:48:12.452993", "That File", "etc. 3")
file_list = [mi_0, mi_1, mi_2, mi_3]

# file_list.sort()
# pprint(file_list)


@total_ordering
@dataclass(frozen=True)
class MultiItem:
    data_source: str
    timestamp: Optional[float]
    creation_date: Optional[str]
    name: str
    owner_etc: str

    def __lt__(self, other: "MultiItem") -> bool:
        ...

    def __eq__(self, other: object) -> bool:
        return self.datetime == cast(MultiItem, other).datetime

    @property
    def datetime(self) -> datetime.datetime:
        if self.data_source == "Local":
            return datetime.datetime.fromtimestamp(
                cast(float, self.timestamp)
            )
        else:
            return datetime.datetime.fromisoformat(
                cast(str, self.creation_date)
            )


@dataclass(frozen=True)
class SimpleMultiItem:
    data_source: str
    timestamp: Optional[float]
    creation_date: Optional[str]
    name: str
    owner_etc: str


def by_timestamp(item: SimpleMultiItem) -> datetime.datetime:
    if item.data_source == "Local":
        return datetime.datetime.fromtimestamp(
            cast(float, item.timestamp)
        )
    elif item.data_source == "Remote":
        return datetime.datetime.fromisoformat(
            cast(str, item.creation_date)
         )
    else:
        raise ValueError(f"Unknown data_source in {item!r}")


# file_list.sort(key=by_timestamp)

# file_list.sort(key=lambda item: item.name)
file_list.sort(key=operator.attrgetter("name"))


song_library = [
    ("Phantom Of The Opera", "Sarah Brightman"),
    ("Knocking On Heaven's Door", "Guns N' Roses"),
    ("Captain Nemo", "Sarah Brightman"),
    ("Patterns In The Ivy", "Opeth"),
    ("November Rain", "Guns N' Roses"),
    ("Beautiful", "Sarah Brightman"),
    ("Mal's Song", "Vixy and Tony"),
]
artists = set()
for song, artist in song_library:
    artists.add(artist)

# print(artists)
# print("Opeth" in artists)
alphabetical = list(artists)
alphabetical.sort()
# print(alphabetical)
#
# for artist in artists:
#     print(f"{artist} plays good music")

# dusty_artists = {
#     "Sarah Brightman",
#     "Guns N' Roses",
#     "Opeth",
#     "Vixy and Tony"
# }
# steve_artists = {"Yes", "Guns N' Roses", "Genesis"}
# print(f"All: {dusty_artists | steve_artists}")
# print(f"Both: {dusty_artists.intersection(steve_artists)}")
# print(f"Either but not both: {dusty_artists ^ steve_artists}")
artists = {"Guns N' Roses", "Vixy and Tony", "Sarah Brightman", "Opeth"}
bands = {"Opeth", "Guns N' Roses"}
print(artists.issuperset(bands))
print(artists.issubset(bands))
print(artists - bands)
print(bands.issuperset(artists))
print(bands.issubset(artists))
print(bands.difference(artists))


class ListQueue(List[Path]):
    def put(self, item: Path) -> None:
        self.append(item)

    def get(self) -> Path:
        return self.pop(0)

    def empty(self) -> bool:
        return len(self) == 0


class DeQueue(Deque[Path]):
    def put(self, item: Path) -> None:
        self.append(item)

    def get(self) -> Path:
        return self.popleft()

    def empty(self) -> bool:
        return len(self) == 0


if TYPE_CHECKING:
    BaseQueue = queue.Queue[Path]  # for mypy
else:
    BaseQueue = queue.Queue      # used at runtime


class ThreadQueue(BaseQueue):
    pass


PathQueue = Union[ListQueue, DeQueue, ThreadQueue]


