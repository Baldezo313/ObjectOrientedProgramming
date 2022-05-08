from __future__ import annotations
from pathlib import Path
from abc import ABC, abstractmethod
# from PIL import Image
from decimal import Decimal
from math import hypot
from urllib.request import urlopen
import time
import fnmatch
import re
import zipfile

from mypy.typeshed.stdlib.tkinter import Image

from model1 import Sample

from database import Database
from typing import List, Optional, Protocol, Any, NoReturn, Union, Tuple, Iterable, cast
from pprint import pprint

#
# class WebPage:
#     def __init__(self, url: str) -> None:
#         self.url = url
#         self._content: Optional[bytes] = None
#
#     @property
#     def content(self) -> bytes:
#         if self._content is None:
#             print("Retrieving New Page...")
#             with urlopen(self.url) as response:
#                 self._content = response.read()
#         return self._content
#
#
# webpage = WebPage("http://ccphillips.net/")
# now = time.perf_counter()
# content1 = webpage.content
# first_fetch = time.perf_counter() - now
# now = time.perf_counter()
# content2 = webpage.content
# second_fetch = time.perf_counter() - now
# assert content2 == content1, "Problem: Pages were different"

# print(f"Initial Request     {first_fetch:.5f}")
# print(f"Subsequent Requests {second_fetch:.5f}")


# class AverageList(List[int]):
#     @property
#     def average(self) -> float:
#         return sum(self) / len(self)
#
#
# a = AverageList([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
# print(a.average)


# class ZipReplace:
#     def __init__(
#             self,
#             archive: Path,
#             pattern: str,
#             find: str,
#             replace: str
#     ) -> None:
#         self.archive_path = archive
#         self.pattern = pattern
#         self.find = find
#         self.replace = replace
#
#     def find_and_replace(self) -> None:
#         input_path, output_path = self.make_backup()
#         with zipfile.ZipFile(output_path, "w") as output:
#             with zipfile.ZipFile(input_path) as input:
#                 self.copy_and_transform(input, output)
#
#     def make_backup(self) -> tuple[Path, Path]:
#         input_path = self.archive_path.with_suffix(
#             f"{self.archive_path.suffix}.old"
#         )
#         output_path = self.archive_path
#         self.archive_path.rename(input_path)
#         return input_path, output_path
#
#     def copy_and_transform(self, input: zipfile.ZipFile, output: zipfile.ZipFile) -> None:
#         for item in input.infolist():
#             extracted = Path(input.extract(item))
#             if not item.is_dir() and fnmatch.fnmatch(item.filename, self.pattern):
#                 print(f"Transform {item}")
#                 input_text = extracted.read_text()
#                 output_text = re.sub(self.find, self.replace, input_text)
#                 extracted.write_text(output_text)
#             else:
#                 print(f"Ignore {item}")
#             output.write(extracted, item.filename)
#             extracted.unlink()
#             for parent in extracted.parents:
#                 if parent == Path.cwd():
#                     break
#                 parent.rmdir()
#
#
# sample = Sample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa")
# if __name__ == "__main__":
#     sample_zip = Path("C:/Users/balde/Desktop/Oreilli/OOP1/archive (1).zip")
#     zr = ZipReplace(sample_zip, "*.md", "xyzzy", "plover's egg")
#     zr.find_and_replace()


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


# class ImgTweaker(ZipProcessor):
#     def transform(self, extracted: Path) -> None:
#         image = Image.open(extracted)
#         scaled = image.resize(size=(640, 960))
#         scaled.save(extracted)
