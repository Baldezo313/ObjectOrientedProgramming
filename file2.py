from __future__ import annotations
from pathlib import Path
from decimal import Decimal
from math import hypot

from database import Database
from typing import List, Optional, Protocol, Any, NoReturn, Union, Tuple, Iterable
from pprint import pprint


# db: Optional[Database] = None
#
#
# def initialize_database(connection):
#     global db
#     db = Database(connection)
#
#
# def get_database(connection: Optional[str] = None) -> Database:
#     global db
#     if not db:
#         db = Database(connection)
#     return db
#
#
# class Formatter:
#     def format(self, string: str) -> str:
#         pass
#
#
# def format_string(string: str, formatter: Optional[Formatter] = None) -> str:
#     """
#     Format a string using the formatter object, which
#     is expected to have a format() method that accept a string
#     """
#     class DefaultFormatter(Formatter):
#         """Format a string in title case."""
#         def format(self, string: str) -> str:
#             return str(string).title()
#     if not formatter:
#         formatter = DefaultFormatter()
#         return formatter.format(string)
#
#
# hello_string = "hello world, how are you today?"
# print(f" input: {hello_string}")
# print(f"output: {format_string(hello_string)}")


# class Contact:
#     all_contacts: List["Contact"] = []
#
#     def __init__(self, name: str, email: str) -> None:
#         self.name = name
#         self.email = email
#         Contact.all_contacts.append(self)
#
#     def __repr__(self) -> str:
#         return (
#             f"{self.__class__.__name__}("
#             f"{self.name!r}, {self.email!r}"
#             f")"
#         )
#
#
# c_1 = Contact("Dusty", "dusty@example.com")
# c_2 = Contact("Steve", "steve@itmaybeahack.com")
# # print(Contact.all_contacts)
#
#
# class Supplier(Contact):
#     def order(self, order: "Order") -> None:
#         print(
#             "If this were a real system we would send "
#             f"'{order}' order to '{self.name}'"
#         )
#
#
# c = Contact("Some Body", "somebody@example.net")
# s = Supplier("Sup Plier", "supplier@example.net")
# # print(c.name, c.email, s.name, s.email)
# # pprint(c.all_contacts)
# # print(c.order("I need pliers")
# print(s.order("I need plier"))

#
# class ContactList(list["Contact"]):
#     def search(self, name: str) -> list["Contact"]:
#         matching_contacts: list["Contact"] = []
#         for contact in self:
#             if name in contact.name:
#                 matching_contacts.append(contact)
#         return matching_contacts
#
#
# class Contact:
#     all_contacts = ContactList()
#
#     def __init__(self, name: str, email: str) -> None:
#         self.name = name
#         self.email = email
#         Contact.all_contacts.append(self)
#
#     def __repr__(self) -> str:
#         return (
#             f"{self.__class__.__name__}("
#             f"{self.name!r}, {self.email!r}" f")"
#         )
#
#
# c1 = Contact("John A", "johna@example.net")
# c2 = Contact("John B", "johnb@sloop.net")
# c3 = Contact("jenna C", "cutty@sark.io")
# print([c.name for c in Contact.all_contacts.search("John")])


# class LongNameDict(dict[str, int]):
#     def longest_key(self) -> Optional[str]:
#         """In effect, max(self, key=len), but less obscure"""
#         longest = None
#         for key in self:
#             if longest is None or len(key) > len(longest):
#                 longest = key
#         return longest


# articles_read = LongNameDict()
# articles_read['lucy'] = 42
# articles_read['c_c_phillips'] = 6
# articles_read['steve'] = 7
# # print(articles_read.longest_key())
# print(max(articles_read, key=len))


# class Friend(Contact):
#     def __init__(self, name: str, email: str, phone: str) -> None:
#         super().__init__(name, email)
#         self.phone = phone
#
#
# f = Friend("Dusty", "Dusty@private.com", "555-1212")
# print(Contact.all_contacts)
#
#
# class Emailable(Protocol):
#     email: str
#
#
# class MailSender(Emailable):
#     def send_mail(self, message: str) -> None:
#         print(f"Sending mail to {self.email=}")
#
#
# class EmailableContact(Contact, MailSender):
#     pass
#
#
# e = EmailableContact("John B", "johnb@sloop.net")
# # print(Contact.all_contacts)
# # print(e.send_mail("Hello, test e-mail here"))
#
#
# class AddressHolder:
#     def __init__(self, street: str, city:str, state:str, code:str) -> None:
#         self.street = street
#         self.city = city
#         self.state = state
#         self.code = code
#
#
# class Friend(Contact, AddressHolder):
#     def __init__(
#         self,
#         name: str,
#         email: str,
#         phone: str,
#         street: str,
#         city: str,
#         state: str,
#         code: str,
#     ) -> None:
#         Contact.__init__(self, name, email)
#         AddressHolder.__init__(self, street, city, state, code)
#         self.phone = phone
#
#
# class BaseClass:
#     num_base_calls = 0
#
#     def call_me(self) -> None:
#         print("Calling method on BaseClass")
#         self.num_base_calls += 1
#
#
# class LeftSubClass(BaseClass):
#     num_left_calls = 0
#
#     def call_me(self) -> None:
#         BaseClass.call_me(self)
#         print("Calling method of LeftSubclass")
#         self.num_left_calls += 1
#
#
# class RightSubclass(BaseClass):
#     num_right_calls = 0
#
#     def call_me(self) -> None:
#         BaseClass.call_me(self)
#         print("Calling method on RightSubclass")
#         self.num_right_calls += 1
#
#
# class Subclass(LeftSubClass, RightSubclass):
#     num_sub_calls = 0
#
#     def call_me(self) -> None:
#         LeftSubClass.call_me(self)
#         RightSubclass.call_me(self)
#         print("Calling method on Subclass")
#         self.num_sub_calls += 1
#
#
# s = Subclass()
# print(s.call_me())
# print(s.num_sub_calls, s.num_left_calls, s.num_right_calls, s.num_base_calls)

#
# class BaseClass:
#     num_base_calls = 0
#
#     def call_me(self):
#         print("Calling method on Base Class")
#         self.num_base_calls += 1
#
#
# class LeftSubclassS(BaseClass):
#     num_left_calls = 0
#
#     def call_me(self) -> None:
#         super().call_me()
#         print("Calling method on LeftSubclass_S")
#         self.num_left_calls += 1
#
#
# class RightSubclassS(BaseClass):
#     num_right_calls = 0
#
#     def call_me(self) -> None:
#         super().call_me()
#         print("Calling method on RightSubclass_S")
#         self.num_right_calls += 1
#
#
# class SubclassS(LeftSubclassS, RightSubclassS):
#     num_sub_calls = 0
#
#     def call_me(self) -> None:
#         super().call_me()
#         print("Calling method on Subclass_S")
#         self.num_sub_calls += 1


# ss = SubclassS()
# print(ss.call_me())
# print(ss.num_sub_calls, ss.num_left_calls, ss.num_right_calls, ss.num_base_calls)

# pprint(SubclassS.__mro__)

#
# class ContactList(list["Contact"]):
#     def search(self, name: str) -> list["Contact"]:
#         matching_contacts: list["Contact"] = []
#         for contact in self:
#             if name in contact.name:
#                 matching_contacts.append(contact)
#         return matching_contacts
#
#
# class Contact:
#     all_contacts = ContactList()
#
#     def __init__(self, /, name: str = "", email:str="", **kwargs: Any) -> None:
#         super().__init__(**kwargs)
#         self.name = name
#         self.email = email
#         self.all_contacts.append(self)
#
#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}(" f"{self.name!r}, {self.email!r}" f")"
#
#
# class AddressHolder:
#     def __init__(
#         self,
#         /,
#         street: str="",
#         city: str="",
#         state:str="",
#         code:str="",
#         **kwargs: Any,
#     ) -> None:
#         super().__init__(**kwargs)
#         self.street = street
#         self.city = city
#         self.state = state
#         self.code = code
#
#
# class Friend(Contact, AddressHolder):
#     def __init__(self, /, phone: str="", **kwargs: Any) -> None:
#         super().__init__(**kwargs)
#         self.phone = phone
#

#
# class AudioFile:
#     ext: str
#
#     def __init__(self, filepath: Path) -> None:
#         if not filepath.suffix == self.ext:
#             raise ValueError("Invalid file format")
#         self.filepath = filepath
#
#
# class MP3File(AudioFile):
#     ext = ".mp3"
#
#     def play(self) -> None:
#         print(f"playing {self.filepath} as mp3")
#
#
# class WavFile(AudioFile):
#     ext = ".wav"
#
#     def play(self) -> None:
#         print(f"playing {self.filepath} as wav")
#
#
# class OggFile(AudioFile):
#     ext = ".ogg"
#
#     def play(self) -> None:
#         print(f"playing {self.filepath} as ogg")
#
#
# p_1 = MP3File(Path("Heart of the Sunrise.mp3"))
# print(p_1.play())
# p_2 = WavFile(Path("Roundabout.wav"))
# print(p_2.play())
# p_3 = OggFile(Path("Heart of the Sunrise.ogg"))
# print(p_3.play())
# p_4 = MP3File(Path("The Fish.mov"))
# print(p_4.play())

#
# class FlacFile:
#     def __init__(self, filepath: Path) -> None:
#         if not filepath.suffix == ".flac":
#             raise ValueError("Not a .flac file")
#         self.filepath = filepath
#
#     def play(self) -> None:
#         print(f"playing {self.filepath} as flac")
#
#
# class EvenOnly(List[int]):
#     def append(self, value: int) -> None:
#         if not isinstance(value, int):
#             raise TypeError("Only integer can be added")
#         if value % 2 != 0:
#             raise ValueError("Only even numbers can be added")
#         super().append(value)
#
#
# e = EvenOnly()
# # e.append("a string")
# # e.append(3)
# # e.append(2)
#
#
# def never_returns() -> NoReturn:
#     print("I am about to raise an exception")
#     raise Exception("This is always raised")
#     print("This line will never execute")
#     return "I won't be returned"
#
#
# # print(never_returns())
#
#
# def call_exception() -> None:
#     print("call_exceptor starts here...")
#     never_returns()
#     print("an exception was raised...")
#     print("...so these lines don't run")
#
#
# # print(call_exception())
#
#
# def handler() -> None:
#     try:
#         never_returns()
#         print("Never executed")
#     except Exception as ex:
#         print(f"I caught an exception: {ex!r}")
#     print("Executed after the exception")
#
#
# # print(handler())
#
#
# def funny_division(divisor: float) -> Union[str, float]:
#     try:
#         return 100 / divisor
#     except ZeroDivisionError:
#         return "Zero is not a good idea!"
#
#
# # print(funny_division(0))
# # print(funny_division(50.0))
# # print(funny_division("hello"))
#
#
# def funnier_division(divisor: int) -> Union[str, float]:
#     try:
#         if divisor == 13:
#             raise ValueError("13 is an unlucky number")
#         return 100 / divisor
#     except (ZeroDivisionError, TypeError):
#         return "Enter a number other than zero"
#

# for val in (0, "hello", 50.0, 13):
#     print(f"Testing {val!r}:", end=" ")
#     print(funnier_division(val))

#
# def funniest_division(divisor: int) -> Union[str, float]:
#     try:
#         if divisor == 13:
#             raise ValueError("13 is an unlucky number")
#         return 100 / divisor
#     except ZeroDivisionError:
#         return "Enter a number other than zero"
#     except TypeError:
#         return "Enter a numerical value"
#     except ValueError:
#         print("No, No, not 13!")
#         raise
#
#
# for val in (0, "hello", 50.0, 13):
#     print(f"Testing {val!r}:", end=" ")
#     print(funniest_division(val))


# some_exceptions = [ValueError, TypeError, IndexError, None]
# for choice in some_exceptions:
#     try:
#         print(f"\nRaising {choice}")
#         if choice:
#             raise choice("An error")
#         else:
#             print("no exception raised")
#     except ValueError:
#         print("Caught a ValueError")
#     except TypeError:
#         print("Caught a TypeError")
#     except Exception as e:
#         print(f"Caught some other error: {e.__class__.__name__}")
#     else:
#         print("This code called if there is no exception")
#     finally:
#         print("This cleanup code is always called")


# class InvalidWithdrawal(ValueError):
#     pass
#
#
# raise InvalidWithdrawal("You don't have $50 in your account")

#
# class InvalidWithdrawal(ValueError):
#     def __init__(self, balance: Decimal, amount: Decimal) -> None:
#         super().__init__(f"account doesn't have ${amount}")
#         self.amount = amount
#         self.balance = balance
#
#     def overage(self) -> Decimal:
#         return self.amount - self.balance
#
#
# # raise InvalidWithdrawal(Decimal('25.00'), Decimal('50.00'))
#
# try:
#     balance = Decimal('25.00')
#     raise InvalidWithdrawal(balance, Decimal('50.00'))
# except InvalidWithdrawal as ex:
#     print("I'm sorry, but your withdrawal is "
#           "more than your balance by "
#           f"${ex.overage()}")


# def divide_with_exception(dividend: int, divisor: int) -> None:
#     try:
#         print(f"{dividend / divisor=}")
#     except ZeroDivisionError:
#         print("You can't divide by zero")
#
#
# def divide_with_if(dividend: int, divisor: int) -> None:
#     if divisor == 0:
#         print("You can't divide by zero")
#     else:
#         print(f"{dividend / divisor=}")

#
# class OutOfStock(Exception):
#     pass
#
#
# class InvalidItemType(Exception):
#     pass
#
#
# class ItemType:
#     def __init__(self, name:str) -> None:
#         self.name = name
#         self.on_hand = 0
#
#
# class Inventory:
#     def __init__(self, stock: list[ItemType]) -> None:
#         pass
#
#     def lock(self, item_type: ItemType) -> None:
#         """Context Entry.
#         Lock the item type so nobody else can manipulate the
#         inventory while we're working."""
#         pass
#
#     def unlock(self, item_type: ItemType) -> None:
#         """Context Exxit.
#         Unlock the item type."""
#         pass
#
#     def purchase(self, item_type: ItemType) -> int:
#         """If the item is not locked, raise a
#         ValueError because something went wrong.
#         If the item_type does not exist,
#         raise InvalidItemType.
#         If the item is currently out of stock,
#         raise OutOfStock.
#         If the item is available,
#         substract one item; return the number of items left.
#         """
#
#         # Mocked results.
#         if item_type.name == "Widget":
#             raise OutOfStock(item_type)
#         elif item_type.name == "Gadget":
#             return 42
#         else:
#             raise InvalidItemType(item_type)
#
#
# widget = ItemType("Widget")
# gadget = ItemType("Gadget")
# inv = Inventory([widget, gadget])
# item_to_buy = widget
# inv.lock(item_to_buy)
#
# try:
#     num_left = inv.purchase(item_to_buy)
# except InvalidItemType:
#     print(f"Sorry, we don't sell {item_to_buy.name}")
# except OutOfStock:
#     print("Sorry, that item is out of stock")
# else:
#     print(f"Purchase complete. There are {num_left} {item_to_buy.name}s left")
# finally:
#     inv.unlock(item_to_buy)
#
# msg = (
#     f"there is {num_left} {item_to_buy.name} left"
#     if num_left == 1
#     else f"there are {num_left} {item_to_buy.name}s left")
# print(msg)


# square = [(1, 1), (1, 2), (2, 2), (2, 1)]
#
#
# def distance(p_1, p_2):
#     return hypot(p_1[0] - p_2[0], p_1[1] - p_2[1])
#
#
# def perimeter(polygon):
#     pairs = zip(polygon, polygon[1:] + polygon[:1])
#     return sum(
#         distance(p1, p2) for p1, p2 in pairs
#     )
#
#
# print(perimeter(square))

# Point = Tuple[float, float]
#
#
# def distance(p_1: Point, p_2: Point) -> float:
#     return hypot(p_1[0] - p_2[0], p_1[1] - p_2[1])
#
#
# Polygon = List[Point]
#
#
# def perimeter(polygon: Polygon) -> float:
#     pairs = zip(polygon, polygon[1:] + polygon[:1])
#     return sum(distance(p1, p2) for p1, p2 in pairs)


class Point:
    def __init__(self, x:float, y:float) -> None:
        self.x = x
        self.y = y

    def distance(self, other: "Point") -> float:
        return hypot(self.x - other.x, self.y - other.y)


# class Polygon:
#     def __init__(self) -> None:
#         self.vertices: List[Point] = []
#
#     def add_point(self, point: Point) -> None:
#         self.vertices.append(point)
#
#     def perimeter(self) -> float:
#         pairs = zip(
#             self.vertices, self.vertices[1:] + self.vertices[:1]
#         )
#         return sum(p1.distance(p2) for p1, p2 in pairs)
#
#
# square = Polygon()
# square.add_point(Point(1,1))
# square.add_point(Point(1,2))
# square.add_point(Point(2,2))
# square.add_point(Point(2,1))
# print(square.perimeter())
#

class Polygon2:
    def __init__(self, vertices: Optional[Iterable[Point]] = None) -> None:
        self.vertices = list(vertices) if vertices else []

    def perimeter(self) -> float:
        pairs = zip(
            self.vertices, self.vertices[1:] + self.vertices[:1]
        )
        return sum(p1.distance(p2) for p1, p2 in pairs)


# square = Polygon2([Point(1, 1), Point(1, 2), Point(2, 2), Point(2, 1)])
# print(square.perimeter())

Pair = Tuple[float, float]
Point_or_Tuple = Union[Point, Pair]


# class Polygon3:
#     def __init__(self, vertices: Optional[Iterable[Point_or_Tuple]] = None) -> None:
#         self.vertices: List[Point] = []
#         if vertices:
#             for point_or_tuple in vertices:
#                 self.vertices.append(self.make_point(point_or_tuple))
#
#     @staticmethod
#     def make_point(item: Point_or_Tuple) -> Point:
#         return item if isinstance(item, Point) else Point(*item)


# class Color:
#     def __init__(self, rgb_value:int, name:str) -> None:
#         self._rgb_value = rgb_value
#         self._name = name
#
#     def set_name(self, name: str) -> None:
#         self._name = name
#
#     def get_name(self) -> str:
#         return self._name
#
#     def set_rgb_value(self, rgb_value: int) -> None:
#         self._rgb_value = rgb_value
#
#     def get_rgb_value(self) -> int:
#         return self._rgb_value
#
#
# c = Color(0xff000, "bright red")
# print(c.get_name())
# c.set_name("red")
# print(c.get_name())


class ColorPy:
    def __init__(self, rgb_value: int, name: str) -> None:
        self.rgb_value = rgb_value
        self.name = name


# c = ColorPy(0xff000, "bright red")
# print(c.name)
# c.name = "red"
# print(c.name)


# class ColorV:
#     def __init__(self, rgb_value: int, name: str) -> None:
#         self._rgb_value = rgb_value
#         if not name:
#             raise ValueError(f"Invalid name {name!r}")
#         self._name = name
#
#     def set_name(self, name: str) -> None:
#         if not name:
#             raise ValueError(f"Invalid name {name!r}")
#         self._name = name
#
#
# class ColorVP:
#     def __init__(self, rgb_value: int, name: str) -> None:
#         self._rgb_value = rgb_value
#         if not name:
#             raise ValueError(f"Invalid name {name!r}")
#         self._name = name
#
#     def _set_name(self, name: str) -> None:
#         if not name:
#             raise ValueError(f"Invalid name {name!r}")
#         self._name = name
#
#     def _get_name(self) -> str:
#         return self._name
#
#     name = property(_get_name, _set_name)
#
#
# c = ColorVP(0xff0000, "bright red")
# # print(c.name)
# c.name = "Red"
# # print(c.name)
# c.name = ""
# print(c.name)

#
# class NorwegianBlue:
#     def __init__(self, name: str) -> None:
#         self._name = name
#         self._state: str
#
#     def _get_state(self) -> str:
#         print(f"Getting {self._name}'s State")
#         return self._state
#
#     def _set_state(self, state: str) -> None:
#         print(f"Setting {self._name}'s State to {state!r}")
#         self._state = state
#
#     def _del_state(self) -> None:
#         print(f"{self._name} is pushing up daisies!")
#         del self._state
#
#     silly = property(_get_state, _set_state, _del_state, "This is a silly property")
#
#
# p = NorwegianBlue("Polly")
# p.silly = "Pining for the fjords"
# print(p.silly)
# del p.silly


class NorwegianBlueP:
    def __init__(self, name: str) -> None:
        self._name = name
        self._state: str

    @property
    def silly(self) -> str:
        """This is a silly property"""
        print(f"Getting {self._name}'s State")
        return self._state

    @silly.setter
    def silly(self, state: str) -> None:
        print(f"Setting {self._name}'s State to {state!r}")
        self._state = state

    @silly.deleter
    def silly(self) -> None:
        print(f"{self._name} is pushing up daisies !")
        del self._state
