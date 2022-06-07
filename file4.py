from __future__ import annotations
from contextlib import contextmanager
from pprint import pprint
from threading import Timer
from urllib.request import urlopen
import datetime
import math
import pickle
import json
from math import radians, pi, hypot, cos
from pathlib import Path
from types import TracebackType
from decimal import Decimal
from typing import Any, Optional, List, Dict, TextIO, Callable, cast, Type, Literal, Iterator, Pattern, Match
from urllib.parse import urlparse
import contextlib
import re
import os
import os.path
import subprocess
import sys
import heapq
import time
from dataclasses import dataclass, field

# ======================================================================
"""
The Intersection of Object-Oriented and Fctional Programming
"""


# ==========================================================

#
# class CustomSequence:
#     def __init__(self, args):
#         self._list = args
#
#     def __len__(self):
#         return 5
#
#     def __getitem__(self, index):
#         return f"x{index}"
#
#
# class FunkyBackwards(list):
#     def __reversed__(self):
#         return "BACKWARDS!"
#
#
# generic = [1, 2, 3, 4, 5]
# custom = CustomSequence([6,7, 8, 9, 10])
# funkadelic = FunkyBackwards([11, 12, 13, 14, 15])
# for sequence in generic, custom, funkadelic:
#     print(f"{sequence.__class__.__name__}: ", end="")
#     for item in reversed(sequence):
#         print(f"{item}, ", end="")
#     print()


# with Path("C:/Users/balde/Desktop/Oreilli/OOP1/README.md").open() as source:
#     for index, line in enumerate(source, start=1):
#         print(f"{index:3d}: {line.rstrip()}")


# def no_params():
#     return "hello, world!"
#
#
# print(no_params())
#
# def mandatory_params(x, y, z):
#     return f"{x=}, {y=}, {z=}"
#
#
# a_variable = 42
# print(mandatory_params("a string", a_variable, True))

# def mandatory_params(x: Any, y: Any, z: Any) -> str:
#     return f"{x=}, {y=}, {z=}"
#
#
# a_variable = 42
# print(mandatory_params("a string", a_variable, True))

# def latitude_dms(
#     deg: float, min: float, sec: float = 0.0, dir: Optional[str] = None
# ) -> str:
#     if dir is None:
#         dir = "N"
#     return f"{deg:02.0f}° {min+sec/60:05.3f}{dir}"
#
#
# print(latitude_dms(36, 51, 2.9, "N"))
# print(latitude_dms(38, 58, dir="N"))
#
# print(latitude_dms(38, 19, dir="N", sec=7))

def kw_only(
        x: Any, y: str = "defaultkw", *, a: bool, b: str = "only"
) -> str:
    return f"{x=}, {y=}, {a=}, {b=}"


# print(kw_only("x"))
# print(kw_only("x", "y", "a"))
# print(kw_only("x", a="a", b="b"))


def pos_only(x: Any, y: str, /, z: Optional[Any] = None) -> str:
    return f"{x=}, {y=}, {z=}"


# print(pos_only(x=2, y="three"))
# print(pos_only(2, "three"))
# print(pos_only(2, "three", 3.14159))


# number = 5
#
#
# def funky_function(x: int = number) -> str:
#     return f"{x=}, {number=}"
#
#
# # print(funky_function(42))
# # number = 7
# # print(funky_function())
#
#
# def better_function(x: Optional[int] = None) -> str:
#     if x is None:
#         x = number
#     return f"better: {x=}, {number=}"
#
#
# # print(better_function(42))
# # number = 7
# # print(better_function())
#
#
# def better_function2(x: Optional[int] = None) -> str:
#     x = number if x is None else x
#     return f"better: {x=}, {number=}"
#
#
# # print(better_function2(42))
# # number = 7
# # print(better_function2())
#
#
# def bad_default(tag: str, history: list[str] = []) -> list[str]:
#     """A Very Bad Design (VBD)."""
#     history.append(tag)
#     return history
#
#
# h = bad_default("tag1")
# h = bad_default("tag2", h)
# # print(h)
# h2 = bad_default("tag21")
# h2 = bad_default("tag22", h2)
# # print(h2)
# # print(h)
# # print(h is h2)
#
#
# def good_default(
#     tag: str, history: Optional[list[str]] = None
# ) -> list[str]:
#     history = [] if history is None else history
#     history.append(tag)
#     return history
#
#
# hh = good_default("tag1")
# hh = good_default("tag2", hh)
# # print(hh)
# hh2 = good_default("tag21")
# hh2 = good_default("tag22", hh2)
# # print(hh2)
# # print(hh)
# # print(hh is hh2)

#
# def get_pages(*links: str) -> None:
#     for link in links:
#         url = urlparse(link)
#         name = "index.html" if url.path in ("", "/") else url.path
#         target = Path(url.netloc.replace(".", "_")) / name
#         print(f"Create {target} from {link!r}")
#         # etc
#
#
# # print(get_pages())
#
# # print(get_pages("https://www.archlinux.org"))
# # print(get_pages('https://www.archlinux.org', 'https://dusty.phillips.codes','https://itmaybehack.com'))
#
#
# class Options(Dict[str, Any]):
#     default_options: dict[str, Any] = {
#         "port": 21,
#         "host": "localhost",
#         "username": None,
#         "password": None,
#         "debug": False,
#     }
#
#     def __init__(self, **kwargs: Any) -> None:
#         super().__init__(self.default_options)
#         self.update(kwargs)
#
#
# options = Options(username="baldezo", password="Onepiece2", debug=True)
# # print(options['debug'])
# # print(options['port'])
# # print(options['username'])
#
#
# def doctest_everything(
#         output: TextIO,
#         *directories: Path,
#         verbose: bool = False,
#         **stems: str
# ) -> None:
#     def log(*args: Any, **kwargs: Any) -> None:
#         if verbose:
#             print(*args, **kwargs)
#         with contextlib.redirect_stdout(output):
#             for directory in directories:
#                 log(f"Searching {directory}")
#                 for path in directory.glob("**/*.md"):
#                     if any(
#                         parent.stem == ".tox"
#                         for parent in path.parents
#                     ):
#                         continue
#                     log(
#                         f"File {path.relative_to(directory)}, "
#                         f"{path.stem=}"
#                     )
#                     if stems.get(path.stem, "").upper() == "SKIPE":
#                         log("Skipped")
#                         continue
#                     options = []
#                     if stems.get(path.stem, "").upper() == "ELLIPSIS":
#                         options += ['ELLIPSIS']
#                     search_path = directory / "src"
#                     print(
#                         f"cd '{Path.cwd()}'; "
#                         f"PYTHONPATH='{search_path}' doctest '{path}' -v"
#                     )
#                     option_args = (
#                         ["-o", ",".join(options)]
#                     )
#                     subprocess.run(
#                         ["python3", "-m", "doctest", "-v"]
#                             + option_args + [str(path)],
#                         cwd=directory,
#                         env={"PYTHONPATH": str(search_path)},
#                     )
#
#
# doctest_everything(sys.stdout,
#                          Path.cwd() / "ch_02",
#                          Path.cwd() / "ch_03")
#
# doctest_log = Path("doctest.log")
# with doctest_log.open('w') as log:
#     doctest_everything(
#         log,
#         Path.cwd() / "ch04",
#         Path.cwd() / "ch_05",
#         verbose=True)
#
# doctest_everything(
#     sys.stdout,
#     Path.cwd() / "ch_02",
#     Path.cwd() / "ch_03",
#     examples="ELLIPSIS",
#     examples_38="SKIP",
#     case_study_2="SKIP",
#     case_study_3="SKIP",
# )


def show_args(arg1, arg2, arg3="THREE"):
    return f"{arg1=}, {arg2=}, {arg3=}"


# some_args = range(3)
# print(show_args(*some_args))

# more_args = {
#     "arg1": "ONE",
#     "arg2": "TWO"
# }
# print(show_args(**more_args))


# def fizz(x: int) -> bool:
#     return x % 3 == 0
#
#
# def buzz(x: int) -> bool:
#     return x % 5 == 0
#
#
# def name_or_number(
#         number: int, *tests: Callable[[int], bool]) -> None:
#     for t in tests:
#         if t(number):
#             return t.__name__
#     return str(number)
#
#
# for i in range(1, 11):
#     print(name_or_number(i, fizz, buzz))


# Callback = Callable[[int], None]
#
#
# @dataclass(frozen=True, order=True)
# class Task:
#     scheduled: int
#     callback: Callback = field(compare=False)
#     delay: int = field(default=0, compare=False)
#     limit: int = field(default=1, compare=False)
#
#     def repeat(self, current_time: int) -> Optional["Task"]:
#         if self.delay > 0 and self.limit > 2:
#             return Task(
#                 current_time + self.delay,
#                 cast(Callback, self.callback),
#                 self.delay,
#                 self.limit -1,
#             )
#         elif self.delay > 0 and self.limit == 2:
#             return Task(
#                 current_time + self.delay,
#                 cast(Callback, self.callback)
#             )
#         else:
#             return None
#
#
# class Scheduler:
#     def __init__(self) -> None:
#         self.tasks: List[Task] = []
#
#     def enter(
#         self,
#         after: int,
#         task: Callback,
#         delay: int = 0,
#         limit: int = 1,
#     ) -> None:
#         new_task = Task(after, task, delay, limit)
#         heapq.heappush(self.tasks, new_task)
#
#     def run(self) -> None:
#         current_time = 0
#         while self.tasks:
#             next_task = heapq.heappop(self.tasks)
#             if (delay := next_task.scheduled - current_time) > 0:
#                 time.sleep(next_task.scheduled - current_time)
#             current_time = next_task.scheduled
#             next_task.callback(current_time)
#             if again := next_task.repeat(current_time):
#                 heapq.heappush(self.tasks, again)
#
#
# def format_time(message: str) -> None:
#     now = datetime.datetime.now()
#     print(f"{now:%I:%M:%S}: {message}")
# #
# #
# # def one(timer: float) -> None:
# #     format_time("Called One")
# #
# #
# # def two(timer: float) -> None:
# #     format_time("Called Two")
# #
# #
# # def three(timer: float) -> None:
# #     format_time("Called Three")
# #
# #
# # class Repeater:
# #     def __init__(self) -> None:
# #         self.count = 0
# #
# #     def four(self, timer: float) -> None:
# #         self.count += 1
# #         format_time(f"Called Four: {self.count}")
# #
# #
# # s = Scheduler()
# # s.enter(1, one)
# # s.enter(2, one)
# # s.enter(2, two)
# # s.enter(4, two)
# # s.enter(3, three)
# # s.enter(6, three)
# # repeater = Repeater()
# # s.enter(5, repeater.four, delay=1, limit=5)
# # s.run()
#
# #
# # class A:
# #     def show_something(self):
# #         print("My class is A")
# #
# #
# # a_object = A()
# # # a_object.show_something()
# #
# #
# # def patched_show_something():
# #     print("My class is NOT A")
# #
# #
# # a_object.show_something = patched_show_something
# # # a_object.show_something()
# # b_object = A()
# # b_object.show_something()
#
#
# class Repeater2:
#     def __init__(self) -> None:
#         self.count = 0
#
#     def __call__(self, timer: float) -> None:
#         self.count += 1
#         format_time(f"Called Four: {self.count}")
#
#
# rpt = Repeater2()
# # rpt(1)
# # rpt(2)
# # rpt(3)
#
# s2 = Scheduler()
# s2.enter(5, Repeater2(), delay=1, limit=5)
# s2.run()

# contents = "Some file contents\n"
# file = open("filename.txt", "w")
# file.write(contents)
# file.close()

# with open("filename.txt") as input:
#     for line in input:
#         print(line)

# results = str(2**2048)
# with open("filename.txt", "w") as output:
#     output.write("# A big number\n")
#     output.writelines(
#         [
#             f"{len(results)}\n",
#             f"{results}\n"
#         ]
#     )

# source_path = Path("filename.txt")
# with source_path.open() as source_file:
#     for line in source_file:
#         print(line, end='')


class StringJoiner(list):
    def __enter__(self):
        return self

    def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        self.result = "".join(self)
        return False


# with StringJoiner("Hello") as sj:
#     sj.append(", ")
#     sj.extend("world")
#     sj.append("!")
#
# print(sj.result)

# with StringJoiner("Partial") as sj:
#     sj.append(" ")
#     sj.extend("Results")
#     sj.append(str(2 / 0))
#     sj.extend("Even If There's an Exception")
#
# print(sj.result)


class StringJoiner2(List[str]):
    def __init__(self, *args: str) -> None:
        super().__init__(*args)
        self.result = "".join(self)


@contextmanager
def joiner(*args: Any) -> Iterator[StringJoiner2]:
    string_list = StringJoiner2(*args)
    try:
        yield string_list
    finally:
        string_list.result = "".join(string_list)


# ===================================================================================================================
"""  
  Strings , Serialization, and File Paths
"""

# s = "Hello world, how are you"
# s2 = s.split(' ')
# print(s.count('l'))
# print(s.find('l'))
# print(s.rindex('m'))
# print(s2)
# print('#'.join(s2))
# print(s.replace(' ', '**'))
# print(s.partition(' '))

# name = "Dusty"
# activity = "reviewing"
# message = f"Hello {name}, you are currently {activity}"
# print(message)

# classname = "MyClass"
# python_code = "print('hello world)"
# template = f"""
# public class {classname} {{
#     public static void main(String[] args){{
#         System.out.println("{python_code}");
#     }}
# }}
# """
# print(template)

emails = ("steve@example.com", "dusty@example.com")
message = {
    "subject": "Next Chapter",
    "message": "Here's the next chapter to review!"
}
# formatted = f"""
# From: <{emails[0]}>
# To: <{emails[1]}>
# Subject: {message['subject']}
#
# {message['message']}
# """


# class Notification:
#     def __init__(
#             self,
#             from_addr: str,
#             to_addr: str,
#             subject: str,
#             message: str
#     ) -> None:
#         self.from_addr = from_addr
#         self.to_addr = to_addr
#         self.subject = subject
#         self._message = message
#
#     def message(self):
#         return self._message
#
#
# email = Notification(
#     "dusty@example.com",
#     "steve@example.com",
#     "Comments on the Chapter",
#     "Can we emphasize Python 3.9 type hints?",
# )
#
# formatted = f"""
# From: <{email.from_addr}>
# To: <{email.to_addr}>
# Subject: {email.subject}
#
# {email.message()}
# """
#
# print(formatted)

# print(f"{[2*a+1 for a in range(5)]}")
# for n in range(1, 5):
#     print(f"{'fizz' if n % 3 == 0 else n}")

# a = 5
# b = 7
# print(f"{a=}, {b=}, {31*a//42*b + b=}")


# def distance(
#         lat1: float, lon1: float, lat2: float, lon2: float
# ) -> float:
#     d_lat = radians(lat2) - radians(lat1)
#     d_lon = min(
#         (radians(lon2) - radians(lon1)) % (2 * pi),
#         (radians(lon1) - radians(lon2)) % (2 * pi),
#     )
#     R = 60 * 180 / pi
#     d = hypot(R * d_lat, R * cos(radians(lat1)) * d_lon)
#     return d
#
#
# annapolis = (38.9784, 76.4922)
# saint_michaels = (38.7854, 76.2233)
# # print(round(distance(*annapolis, *saint_michaels), 9))
# oxford = (38.6865, 76.1716)
# cambridge = (38.5632, 76.0788)
# legs = [
#     ("to st.michaels", annapolis, saint_michaels),
#     ("to oxford", saint_michaels, oxford),
#     ("to cambridge", oxford, cambridge),
#     ("return", cambridge, annapolis),
# ]
#
# speed = 5
# fuel_per_hr = 2.2
# for name, start, end in legs:
#     d = distance(*start, *end)
#     print(name, d, d/speed, d/speed*fuel_per_hr)

# print(f"{'leg':16s} {'dist':5s} {'time':4s} {'fuel':4s}")
# for name, start, end in legs:
#     d = distance(*start, *end)
#     print(
#         f"{name:16s} {d:5.2f} {d/speed:4.1f} "
#         f"{d/speed*fuel_per_hr:4.0f}"
#     )


# important = datetime.datetime(2019, 10, 26, 13, 14)
# print(f"{important:%Y-%m-%d %I:%M%p}")
#
# subtotal = Decimal('2.95') * Decimal('1.0625')
# template = "{label}: {number:*^{size}.2f}"
# print(template.format(label="Amount", size=10, number=subtotal))
#
# grand_total = subtotal + Decimal('12.34')
# print(template.format(label="Total", size=12, number=grand_total))

# print(list(map(hex, b'abc')))
# print(list(map(bin, b'abc')))

# print(bytes([137, 80, 78, 71, 13, 10, 26, 10]))

# characters = b'\x63\x6c\x69\x63\x68\xc3\xa9'
# print(characters)
# print(characters.decode("utf-8"))
# print(characters.decode("iso8859-5"))
# print(characters.decode("cp037"))

characters = "cliché"
# print("UTF-8", characters.encode("UTF-8"))
# print("latin-1", characters.encode("latin-1"))
# print("cp1252", characters.encode("cp1252"))
# print("CP437", characters.encode("CP437"))
# print("ascii", characters.encode("ascii"))
#
# print("replace ascii by replace", characters.encode("ascii", "replace"))
# print("ignore the ascii", characters.encode("ascii", "ignore"))
# print("use XML for the ascii", characters.encode("ascii", "xmlcharrefreplace"))

ba = bytearray(b"abcdefgh")
# ba[4:6] = b"\x15\xa3"
# print(ba)
# ba[3] = ord(b'g')
# ba[4] = 68
# print(ba)

search_string = "hello world"
pattern = r"hello world"
# if match := re.match(pattern, search_string):
#     print("regex matches")
#     print(match)


def matchy(pattern: Pattern[str], text: str) -> None:
    if match := re.match(pattern, text):
        print(f"{pattern=!r} matches at {match=!r}")
    else:
        print(f"{pattern=!r} not found in {text=!r}")


# print(matchy(pattern=r"hello wo", text="hello world"))
# print(matchy(pattern=r"ello world", text="hello world"))

# print(matchy(pattern=r"^hello world$", text="hello world"))
# print(matchy(pattern=r"^hello world$", text="hello worl"))

# print(matchy(pattern=r"\^hello world\$", text="hello worl"))
# print(matchy(pattern=r"\^hello world\$", text="^hello world$"))

# print(matchy(r'\d\d\s\w\w\w\s\d\d\d\d', '26 Oct 2019'))

# print(matchy(r'hel*o', 'hello'))
# print(matchy(r'hel*o', 'heo'))
# print(matchy(r'hel*o', 'helllllo'))

# print(matchy(r'[A-Z][a-z]* [a-z]*\.', "A string."))
# print(matchy(r'[A-Z][a-z]* [a-z]*\.', "No ."))
# print(matchy(r'[a-z]*.*', ""))

# print(matchy(r'\d+\.\d+', "0.4"))
# print(matchy(r'\d+\.d+', "1.002"))
# print(matchy(r'\d+\.\d+', "1."))
# print(matchy(r'\d?\d%', "1%"))
# print(matchy(r'\d?\d%', "99%"))
# print(matchy(r'\d?\d%', "100%"))

# print(matchy(r'[A-Z][a-z]*( [a-z]+)*\.$', "Eat."))
# print(matchy(r'[A-Z][a-z]*( [a-z]+)*\.$', "Eat more good food."))
# print(matchy(r'[A-Z][a-z]*( [a-z]+)*\.$', "A good meal."))


# def email_domain(text: str) -> Optional[str]:
#     email_pattern = r"[a-z0-9._%+-]+@([a-z09.-]+\.[a-z]{2,})"
#     if match := re.match(email_pattern, text, re.IGNORECASE):
#         return match.group(1)
#     else:
#         return None
#
#
# def email_domain_2(text: str) -> Optional[str]:
#     email_pattern = r"(?P<name>[a-z0-9._%+-]+)@(?P<domain>[a-z0-9.-]+\.[a-z]{2,})"
#     if match := re.match(email_pattern, text, re.IGNORECASE):
#         return match.groupdict()["domain"]
#     else:
#         return None


# print(re.findall(r"\d+[hms]", "3h 2m  45s"))
# print(re.findall(r"(\d+)[hms]", "3h:2m:45s"))
# print(re.findall(r"(\d+)([hms])", "3h, 2m, 45s"))
# print(re.findall(r"((\d+)([hms]))", "3h - 2m - 45s"))

# print(re.findall(r"\d+[hms]", "3h 2m  45s"))

duration_pattern = re.compile(r"\d+[hms]")
# print(duration_pattern.findall("3h 2m  45s"))
# print(duration_pattern.findall("3h:2m:45s"))


# path = os.path.abspath(
#     os.sep.join(
#         ["", "Users", "dusty", "subdir", "subsubdir", "file.ext"]
#     )
# )
# print(path)

# path = Path("/Users") / "dusty" / "subdir" / "subsubdir" / "file.ext"
# print(path)


# def scan_python_1(path: Path) -> int:
#     sloc = 0
#     with path.open() as source:
#         for line in source:
#             line = line.strip()
#             if line and not line.startswith("#"):
#                 sloc += 1
#     return sloc
#
#
# def count_sloc(path: Path, scanner: Callable[[Path], int]) -> int:
#     if path.name.startswith("."):
#         return 0
#     elif path.is_file():
#         if path.suffix != ".py":
#             return 0
#         with path.open() as source:
#             return scanner(path)
#     elif path.is_dir():
#         count = sum(
#             count_sloc(name, scanner) for name in path.iterdir()
#         )
#         return count
#     else:
#         return 0
#
#
# base = Path.cwd().parent
# chapter = base / "file1.py"
# count = count_sloc(chapter, scan_python_1)
# print(
#     f"{chapter.relative_to(base)}: {count} lines of code"
# )


# some_data = [
#     "a list", "containing", 5, "items",
#     {"including": ["str", "int", "dict"]}
# ]
# with open("pickled_list", 'wb') as file:
#     pickle.dump(some_data, file)
#
# with open("pickled_list", "rb") as file:
#     loaded_data = pickle.load(file)
#
# print(loaded_data)
#
# assert loaded_data == some_data

#
# class URLPolling:
#     def __init__(self, url: str) -> None:
#         self.url = url
#         self.contents = ""
#         self.last_updated: datetime.datetime
#         self.timer: Timer
#         self.update()
#
#     def update(self) -> None:
#         self.contents = urlopen(self.url).read()
#         self.last_updated = datetime.datetime.now()
#         self.schedule()
#
#     def schedule(self) -> None:
#         self.timer = Timer(3600, self.update)
#         self.timer.setDaemon(True)
#         self.timer.start()
#
#     def __getstate__(self) -> dict[str, Any]:
#         pickleable_state = self.__dict__.copy()
#         if "timer" in pickleable_state:
#             del pickleable_state["timer"]
#         return pickleable_state
#
#     def __setstate__(self, pickleable_state: dict[str, Any]) -> None:
#         self.__dict__ = pickleable_state
#         self.schedule()
#
#
# poll = URLPolling("http://dusty.phillips.codes")
# pprint(pickle.dumps(poll))


class Contact:
    def __init__(self, first, last):
        self.first = first
        self.last = last

    @property
    def full_name(self):
        return ("{} {}".format(self.first, self.last))


# c = Contact("Noriko", "Hannah")
# print(json.dumps(c.__dict__))


class ContactEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Contact):
            return {
                "__class__": "Contact",
                "first": obj.first,
                "last": obj.last,
                "full_name": obj.full_name
            }
        return super().default(obj)


c = Contact("Noriko", "Hannah")
text = json.dumps(c, cls=ContactEncoder)
print(text)
