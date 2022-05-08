# print(type("Hello, World!"))
# print(type(42))
# a_string_variable = "Hello, world!"
# print(type(a_string_variable))
# a_string_variable = 42
# print(type(a_string_variable))

# def odd(n):
#     return n % 2 != 0
#
#
# # print(odd(3))
# print(odd(4))
# print(odd("Hello, world!"))
#
#
# def main():
#     print(odd("Hello, world!"))
#
#
# if __name__ == "__main__":
#     main()


# ========================================================

# class MyFirstClass:
#     pass
#
#
# a = MyFirstClass()
# b = MyFirstClass()
# # print(a)
# # print(b)
# print(a is b)

# class Point:
#     pass
#
#
# p1 = Point()
# p2 = Point()
# p1.x = 5
# p1.y = 4
# p2.x = 3
# p2.y = 6
# print(p1.x, p1.y)
# print(p2.x, p2.y)


# class Point:
#     def reset(self):
#         self.x = 0
#         self.y = 0
# p = Point()
# p.reset()
# print(p.x, p.y)


import math


# class Point:
#     def move(self, x, y):
#         self.x = x
#         self.y = y
#
#     def reset(self):
#         self.move(0, 0)
#
#     def calculate_distance(self, other: "Point"):
#         return math.hypot(self.x - other.x, self.y - other.y)
#
#
# point1 = Point()
# point2 = Point()
# point1.reset()
# point2.move(5, 0)
# print(point2.calculate_distance(point1))
#
# assert point2.calculate_distance(point1) == point1.calculate_distance(point2)
# point1.move(3, 4)
# print(point1.calculate_distance(point2))
# print(point1.calculate_distance(point1))


# class Point:
#     def __init__(self, x, y):
#         self.move(x, y)
#
#     def move(self, x, y):
#         self.x = x
#         self.y = y
#
#     def reset(self):
#         self.move(0, 0)
#
#     def calculate_distance(self, other: "Point"):
#         return math.hypot(self.x - other.x, self.y - other.y)
#
#
# point = Point(3, 5)
# print(point.x, point.y)


class Point:
    """
    Represents a point in two-dimensional geometric coordinates
    p_0 = Point()
    p_1 = Point(3, 4)
    p_0.calculate_distance(p_1)
    5.0
    """
    def __init__(self, x, y):
        """
        Initialize the position of a new point. The x and y
        coordinates can be specified. If they are not, the
        point defaults to the origin.
        :param x: float x-coordinate
        :param y: float y-coordinate
        """
        self.move(x, y)

    def move(self, x, y):
        """
        Move the point to a new location in 2D space.
        :param x: float x-coordinate
        :param y: float y-coordinate
        """
        self.x = x
        self.y = y

    def reset(self):
        """
        Reset the point back to the geometric origin: 0, 0
        """
        self.move(0, 0)

    def calculate_distance(self, other: "Point"):
        """
        Calculate the Euclidean distance from this point
        to a second point passed as a parameter.
        :param other: Point instance
        :return: float distance
        """
        return math.hypot(self.x - other.x, self.y - other.y)
