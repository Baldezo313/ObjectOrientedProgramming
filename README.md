# ***Python Object-Oriented Programming***  

## **0. Preface**  
The python programming language is extremely popular and used for a variety of applications. The Python language is
designed to make it relatively easy to create small programs. To create more sophisticated software,
we need to acquire a number of important programming and software design skills.  
  
This book describes the **object-oriented** approach to creating programs in Python. It 
introduces the terminology of object-oriented programming, demonstrating software design and Python 
programming through step-by-step examples. It describes how to make use of inheritance and composition
to build software from individual elements. It shows how to use Python's built-in exceptions and data 
structures, as well as elements of the Python standard library. A number of common design patterns are 
described with detailed examples.  
  
  
## **1. Object-Oriented Design**  
In software development, design is often considered as the step's done *before* programming. This isn't true;
in reality, analysis, programming, and design tend to overlap, combine, and interweave. Throughout this book,
we'll be covering a mixture of design and programming issues without trying to parse them into separate 
buckets. One of the advantages of a language like Python is the ability to express the design clearly.
In this chapter, we will talk a little about how we can move from a good idea toward writing software. 
We'll create some design artifacts-like diagrams-that can help clarify our thinking before we start writing
code.   
  
### **Recall**  
Some key points in this chapter:  
  
* Analyzing problem requirements in an object-oriented context  
* How to draw **Unified Modeling Language (UML)** diagrams to communicate how the system works  
* Discussing object-oriented system using correct terminology and jargon  
* Understanding the distinction between class, object, attribute, and behavior  
* Some OO design techniques are used more than others. In our case study example, we focused on a few:  
  * Encapsulating features into classes
  * Inheritance to extend a class with features  
  * Composition to build a class from component objects.  
  
### **Summary**  
In this chapter, we took a whirlwind tour through the terminology of the object-oriented paradigm, focusing
on object-oriented design. We can separate different objects into a taxonomy of different class and describe
the attributes and behaviors of those objects via the class interface. Abstraction, encapsulation, and
information hiding are highly-related concepts. There are many kinds of relationship between objects, including
association, composition, and inheritance. UML syntax can be useful for fun and communication.  
  
  
## **2. Objects in Python**  
We have a design in hand and are ready to turn that design into a working program! Of course, it doesn't
usually this way. We'll be seeing examples and hints for good software design throughout the book, but our focus
is object-oriented programming. So let's have a look at the Python syntax that allows us to create object-oriented software.  
  
### **Recall**  
Some key points in this chapter:  
* Python has optional type hints to help describe how data objects are related and what the parameters 
 should be for methods and functions.  
* We create Python classes with the `class` statement. We should initialize the attributes in the special `__init__()` method.  
* Modules and packages are used as higher-level groupings of classes.  
* We need to plan out the organization of module content. While the general advice is "flat is better than nested",
 there are few cases where it can be helpful to have nested packages.  
* Python has no notion of "private" data. We often say "we're all adults here."; we can see the source
 code, and private declarations aren't very helpful. This doesn't change our design; it simply removes
the need for a handful of keywords.  
* We can install third-party packages using PIP tools. We can create a virtual environment, for example, with `venv`


### **Summary**  
In this chapter, we learned how simple it is to create classes and assign properties and methods in Python.
Unlike many languages, Python differentiates between a constructor and initializer. It has a relaxed attitude
toward access control.  
There are many levels of scope, including packages, modules, classes, and functions. We understood the 
difference between relative and absolute imports, and how to manage third-party packages that don't come with Python.  
  
  
## **3. When Objects Are Alike**  
In the programming world, duplicate code is considered evil. We should not have multiple copies of the 
same, or similar, code in different places. When we fix a bug in one copy and fail to fix the same bug
in another copy, we've caused no end of problems for ourselves.  
  
There are many ways to merge pieces of code or objects that have a similar functionality. In this chapter,
we'll be covering the most famous object-oriented principal: inheritance. Inheritance allows us to create
"is-a" relationships between two or more classes, abstracting common logic into superclasses and extending
the superclass with specific details in each subclass.   
  
### **Recall**  
Some key points in this chapter:  
* A central object-oriented design principle is inheritance: a subclass can inherit aspects of a superclass,
saving copy-and-paste programming. A subclass can extend the superclass to add features or specialize the
superclass in other ways.  
* Multiple inheritance is a feature of Python. The most common form is a host class with mixin class definitions. 
We can combine multiple classes leveraging the method resolution order handling common features like initialization.
* Polymorphism lets us create multiple classes that provide alternative implementations for fulfilling a contract.  
 Because of Python's duck typing rules, any classes that have the right methods can substitute for each other.  
  
### **Summary**  
We've gone from simple inheritance, one of the most useful tools in the object-oriented programmer's toolbox, all
the way through to multiple inheritance-one of the most complicated. Inheritance can be used to add functionality
to existing classes and built-in generics. Abstracting similar code into a parent class can help increase 
maintainability. Methods on parent classes can be called using `super`, and argument list must be formatted safely 
for these calls to work when using multiple inheritance.  
  
  
## **4. Expecting the Unexpected**  
Systems built with software can be fragile. While the software is highly predictable, the runtime context can 
provide unexpected inputs and situations. Devices fail, networks are unreliable, mere anarchy is loosed on our
application. We need to have a way to work around the spectrum of failures that plague computer systems.
  
There are two broad approaches to dealing with the unforeseen. One approach is to return a recognizable 
error-signaling value from a function. A value, like `None`, could be used. Other library functions can then be 
used by an application to retrieve details of the erroneous condition. A variation of this theme is to paire a return
from OS request with a success or failure indicator. The other approach is to interrupt the normal, 
sequential execution of statements and divert to statements that handle exceptions. This second approach
is what Python does: it eliminates the need to check return values for errors.  
  
In this chapter, we will study **exceptions**, special error objects raised when a normal response is impossible.  
  
### **Recall**  
Some key points in this chapter:  
* Raising an exception happens when something goes wrong. We looked at division by zero as an example. 
Exceptions can also be raised with the `raise` statement.
* The effects of an exception are to interrupt the normal sequential execution of statements. 
It saves us from having to write a lot of `if` statements to check to see if things can possibly 
work or check to see if something actually failed.  
* Handling exceptions is done with the `try`: statement, 
which has an `except`: clause for each kind of exception we want to handle.
* The exception hierarchy follows object-oriented design patterns to define 
a number of subclasses of the `Exception` class we can work with. Some additional exceptions, 
`SystemExit` and `KeyboardInterrupt`, are not subclasses of the `Exception` class; handling 
introduces risks and doesn't solve very many problems, so we generally ignore them.  
* Defining our own exceptions is a matter of extending the `Exception` class. 
This makes it possible to define exceptions with very specific semantics.

### **Summary**  
In this chapter, we went into the gritty details of raising, handling, defining, and manipulating 
exceptions. Exceptions are a powerful way to communicate unusual circumstances or error conditions 
without requiring a calling function to explicitly check return values. There are many built-in 
exceptions and raising them is trivially easy. 
There are some syntax's for handling different exception events.  
   
## **5. When to Use Object-Oriented Programming**  
In this chapter, we'll discuss some useful applications of the knowledge we've gained, 
looking at some new topics along the way:

* How to recognize objects
* Data and behaviors, once again
* Wrapping data behaviors using properties
* The Don't Repeat Yourself principle and avoiding repetition.  
  
### **Recall**  
Here are some of the key points in this chapter:
* When we have both data and behavior, this is the sweet spot for object-oriented design. 
We can leverage Python's generic collections and ordinary functions for many things. 
When it becomes complex enough that we need to be sure that pieces are all defined together, 
then we need to start using classes.  
* When an attribute value is a reference to another object, the Pythonic approach is to allow 
direct access to the attribute; we don't write elaborate setter and getter functions. 
When an attribute value is computed, we have two choices: we can compute it eagerly or lazily. 
A property lets us be lazy and do the computation just in time.  
* We'll often have cooperating objects; the behavior of the application emerges from the cooperation. 
This can often lead to manager objects that combine behaviors from component class definitions to create an integrated, working whole.

### **Summary**  
In this chapter, we focused on identifying objects, especially objects that are not immediately 
apparent; objects that manage and control. Objects should have both data and behaviors, 
but properties can be used to blur the distinction between the two. The DRY principle is 
an important indicator of code quality, and inheritance and composition can be applied 
to reduce code duplication.  
  
## **6. Abstract Base Classes and Operator Overloading**  
We often need to make a distinction between concrete classes that have a complete 
set of attributes and methods, and an abstract class that is missing some details. 
This parallels the philosophical idea of abstraction as a way to summarize complexities. 
We might say that a sailboat and an airplane have a common, abstract relationship of being 
vehicles, but the details of how they move are distinct.  
  
In Python, we have two approaches to defining similar things:

* **Duck typing**: When two class definitions have the same attributes and methods, 
 then instances of the two classes have the same protocol and can be used interchangeably. 
 We often say, "When I see a bird that walks like a duck and swims like a duck and quacks 
 like a duck, I call that bird a duck."
* **Inheritance**: When two class definitions have common aspects, a subclass can share 
 common features of a superclass. The implementation details of the two classes may vary, 
 but the classes should be interchangeable when we use the common features defined by the superclass.  
  

We can take inheritance one step further. We can have superclass definitions that are abstract: 
this means they aren't directly usable by themselves, but can be used through inheritance 
to create concrete classes.  
  
### **Recall**  
Here are some pf the key points in this chapter:  
* Using abstract base class definitions is a way to create class definitions with placeholders.  
 This is a handy technique, and can be somewhat clearer than using `raise NotImplementedError` in 
 unimplemented methods.  
* ABCs and type hints provide ways to create class definition. An ABC is type hint that can help to clarify
 the essential features we need from an object. It's common, for example, to use `Iterable[X]` to emphasize
 that we need one aspect of a class implementation.  
* The `collections.abc` module defines abstract base classes for Python's built-in collections. 
When we want to make our own unique collect class that can integrate seamlessly with Python, 
we need to start with the definitions from this module.  
* Creating your own abstract base class leverages the `abc` module. 
The `abc.ABC` class definition is often a perfect starting point for creating an abstract base class.
* The bulk of the work is done by the `type` class. 
It's helpful to review this class to understand how classes are created by the methods of `type`.
* Python operators are implemented by special methods in classes. 
We can – in a way – "overload" an operator by defining appropriate special methods 
so that the operator works with objects of our unique class.  
* Extending built-ins is done via a subclass that modifies the behavior of a built-in type. 
We'll often use `super()` to leverage the built-in behavior.
* We can implement our own metaclasses to change – in a 
fundamental way – how Python class objects are built.  
  
### **Summary**  
In this chapter, we focused on identifying objects, especially objects that are not immediately 
apparent; objects that manage and control. Objects should have both data and behaviors, 
but properties can be used to blur the distinction between the two. The DRY principle is 
an important indicator of code quality, and inheritance and composition can be applied 
to reduce code duplication.  
  
## **7. Python Data Structures**  
In this chapter, we'll discuss the object-oriented features of these data structures, 
when they should be used instead of a regular class, and when they should not be used.  
  
### **Recall**  
We've explored a variety of built-in Python data structures in this chapter. 
Python lets us do a great deal of object-oriented programming without the overheads of numerous, 
potentially confusing, class definitions. We can rely on a number of built-in 
classes where they fit our problem.

In this chapter, we looked at the following:
* Tuples and named tuples let us leverage a simple collection of attributes. 
We can extend the `NamedTuple` definition to add methods when those are necessary.
* Dataclasses provide sophisticated collections of attributes. A variety of methods 
can be provided for us, simplifying the code we need to write.
* Dictionaries are an essential feature, used widely in Python. There are many places where keys are associated with values. 
The syntax for using the built-in dictionary class makes it easy to use.
* Lists and sets are also first-class parts of Python; our applications can make use of these.
* We also looked at three types of queues. These are more specialized structures with more focused patterns 
of access than a generic list object. The idea of specialization and narrowing the domain of features 
can lead to performance improvements, also, making the concept widely applicable.  
  
### **Summary**  
We've covered several built-in data structures and attempted to understand how to choose one for specific applications. 
Sometimes, the best thing we can do is create a new class of objects, but often, one of the built-ins 
provides exactly what we need. When it doesn't, we can always use inheritance or composition 
to adapt them to our use cases. We can even override special methods to completely 
change the behavior of built-in syntaxes.  
  
## **8. The Intersection of Object-Oriented and Functional Programming**  
There are many aspects of Python that appear more reminiscent of structural or functional programming 
than object-oriented programming. Although object-oriented programming has been the most visible 
paradigm of the past two decades, the old models have seen a recent resurgence. 
As with Python's data structures, most of these tools are syntactic sugar over an underlying 
object-oriented implementation; we can think of them as a further abstraction layer built on 
top of the (already abstracted) object-oriented paradigm. In this chapter, 
we'll be covering a grab bag of Python features that are not strictly object-oriented.  
  
### **Recall**  
We've touched on a number of ways that object-oriented and functional programming techniques are part of Python:  
* Python built-in functions provide access to special methods that can be implemented by a wide 
variety of classes. Almost all classes, most of them utterly unrelated, provide an implementation 
for `__str__( )` and `__repr__()` methods, which can be used by the built-in `str()` and `repr()` functions. 
There are many functions like this where a function is provided to access implementations that cut across class boundaries. 
* Some object-oriented languages rely on "method overloading" – a single name can have multiple 
implementations with different combinations of parameters. Python provides an alternative, 
where one method name can have optional, mandatory, position-only, and keyword-only parameters. 
This provides tremendous flexibility.  
* Functions are objects and can be used in ways that other objects are used. We can provide them as argument values; 
we can return them from functions. A function has attributes, also.  
* File I/O leads us to look closely at how we interact with external objects. Files are always composed of bytes. 
Python will convert the bytes to text for us. The most common encoding, UTF-8, is the default, but we can specify other encodings.  
* Context managers are a way to be sure that the operating system entanglements are correctly cleaned up even when there's an exception raised. 
The use goes beyond simply handling files and network connections, however. Anywhere we have a clear context where 
we want consistent processing on entry or exit, we have a place where a context manager can be useful.  
  
### **Summary**  
We covered a grab bag of topics in this chapter. Each represented an important non-object-oriented feature that is popular in Python. 
Just because we can use object-oriented principles does not always mean we should!

However, we also saw that Python typically implements such features by providing a syntax shortcut to traditional object-oriented syntax. 
Knowing the object-oriented principles underlying these tools allows us to use them more effectively in our own classes.

We discussed a series of built-in functions and file I/O operations. There are a whole bunch of different syntaxes 
available to us when calling functions with arguments, keyword arguments, and variable argument lists. 
Context managers are useful for the common pattern of sandwiching a piece of code between two method calls. 
Even functions are objects, and, conversely, any normal object can be made callable.  
  
  
## **9. Strings, Serialization, and File Paths**  
Before we get involved with higher-level design patterns, let's take a deep dive into one of Python's most common objects: 
the string. We'll see that there is a lot more to the string than meets the eye, and we'll also cover searching strings 
for patterns, and serializing data for storage or transmission.  
  
We often take persistence – the ability to write data to a file and retrieve it at an arbitrary later date – for granted. 
Because persistence happens via files, at the byte level, via OS writes and reads, it leads 
to two transformations: data we have stored must be decoded into a nice, useful object collection 
of objects in memory; objects from memory need to be encoded to some kind of clunky text or bytes 
format for storage, transfer over the network, or remote invocation on a distant server.  
  
### **Recall**  
In this chapter, we've looked at the following topics:  
* The ways to encode strings into bytes and decode bytes into strings. While some older character encodings (like ASCII) 
treat bytes and characters alike, this leads to confusion. Python text can be any Unicode character
and Python bytes are numbers in the range 0 to 255.  
* String formatting lets us prepare string objects that have template pieces and dynamic pieces. This works for a lot of 
situations in Python. One is to create readable output for people, but we can use f-strings and the string `format()` method 
everywhere we're creating a complex string from pieces.  
* We use regular expressions to decompose complex strings. In effect, a regular expression is the opposite of a fancy string formatter. 
Regular expressions struggle to separate the characters we're matching from "meta-characters" that provide additional matching rules, 
like repetition or alternative choices.
* We've looked at a few ways to serialize data, including Pickle, CSV, and JSON. There are other formats, including YAML, 
that are similar enough to JSON and Pickle that we didn't need to cover them in detail. Other serializations like XML and 
HTML are quite a bit more complex, and we've avoided them.  
  
### **Summary**  
We've covered string manipulation, regular expressions, and object serialization in this chapter. 
Hardcoded strings and program variables can be combined into outputtable strings using the powerful string formatting system. 
It is important to distinguish between binary and textual data, and `bytes` and `str` have specific purposes that must be understood. 
Both are immutable, but the bytearray type can be used when manipulating bytes.

Regular expressions are a complex topic, and we only scratched the surface. There are many ways to
serialize Python data; pickles and JSON are two of the most popular.  
  
  
## **10. The Iterator Pattern**  

We've discussed how many of Python's built-ins and idioms seem, at first blush, to fly in the face
of object-oriented principles, but are actually providing access to real objects under the hood. 
In this chapter, we'll discuss how the `for loop`, which seems so structured, is actually a 
lightweight wrapper around a set of object-oriented principles. We'll also see a variety 
of extensions to this syntax that automatically create even more types of object.  
  
### **Recall**  
This chapter looked at a design pattern that seems ubiquitous in Python, the iterator. 
The Python iterator concept is a foundation of the language and is used widely. 
In this chapter we examined a number of aspects:  
* Design patterns are good ideas we see repeated in software implementations, designs, and architectures. 
A good design pattern has a name, and a context where it's usable. Because it's only a pattern, 
not reusable code, the implementation details will vary each time the pattern is followed.  
* `The Iterator` protocol is one of the most powerful design patterns because it provides a 
consistent way to work with data collections. We can view strings, tuples, lists, sets, and even 
files as iterable collections. A mapping contains a number of iterable collections including 
the keys, the values, and the items (key and value pairs.)
* List, set, and dictionary comprehensions are short, pithy summaries of how to create a new collection 
from an existing collection. They involve a source iterable, an optional filter, and a final 
expression to define the objects in the new collection.  
* Generator functions build on other patterns. They let us define iterable objects that have map and filter capabilities.  
  
### **Summary**  
In this chapter, we learned that design patterns are useful abstractions that provide best-practice
solutions for common programming problems. We covered our first design pattern, the iterator, 
as well as numerous ways that Python uses and abuses this pattern for its own nefarious purposes. 
The original iterator pattern is extremely object-oriented, but it is also rather ugly and verbose
to code around. However, Python's built-in syntax abstracts the ugliness away, leaving us with 
a clean interface to these object-oriented constructs.

Comprehensions and generator expressions can combine container construction with iteration in a single line. 
Generator functions can be constructed using the yield syntax.  
  
## **11. Common Design Patterns**
In the previous chapter, we were briefly introduced to design patterns, and covered the iterator pattern, a pattern 
so useful and common that it has been abstracted into the core of the programming language itself. In this chapter, 
we'll be reviewing other common patterns and how they are implemented in Python. As with iteration, Python often provides 
an alternative syntax to make working with such problems simpler. We will cover both the traditional design, and the Python version for these patterns.
  
### **Recall**  
The world of software design is full of good ideas. The really good ideas get repeated and form repeatable patterns. 
Knowing – and using – these patterns of software design can save the developer from burning a lot of brain calories trying 
to reinvent something that's been developed already. In this chapter, we looked at a few of the most common patterns:  
* The Decorator pattern is used in the Python language to add features to functions or classes. We can define decorator functions 
and apply them directly, or use the `@` syntax to apply a decorator to another function.  
* The Observer pattern can simplify writing GUI applications. It can also be used in non-GUI applications to formalize 
relationships between objects that change state, and objects that display or summarize or otherwise use the state information.  
* The Strategy pattern is central to a lot of object-oriented programming. We can decompose large problems into containers 
with the data and strategy objects that help with processing the data. The Strategy object is a kind of "plug-in" to another object. 
This gives us ways to adapt, extend, and improve processing without breaking all the code we wrote when we make a change.  
* The Command pattern is a handy way to summarize a collection of changes that are applied to other objects. 
It's really helpful in a web services context where external commands arrive from web clients.  
* The State pattern is a way to define processing where there's a change in state and a change in behavior. 
We can often push unique or special-case processing into state-specific objects, leveraging the Strategy pattern to plug in state-specific behavior.
* The Singleton pattern is used in the rare cases where we need to be sure there is one and only one of a specific kind of object. It's common, 
for example, to limit an application to exactly one connection to a central database.  
  

These design patterns help us organize complex collections of objects. Knowing a number of patterns can help the developer 
visualize a collection of cooperating classes, and allocate their responsibilities. It can also help developers talk about a design: 
when they've both read the same books on design patterns, they can refer to the patterns by name and skip over long descriptions   

### **Summary**  
 This chapter discussed several common design patterns in detail, with examples, UML diagrams, and a discussion of 
 the differences between Python and statically typed object-oriented languages. 
 The Decorator pattern is often implemented using Python's more generic decorator syntax. 
 The Observer pattern is a useful way to decouple events from actions taken on those events. 
 The Strategy pattern allows different algorithms to be chosen to accomplish the same task. 
 The Command pattern helps us design active classes that share a common interface but carry out distinct actions. 
 The State pattern looks similar to the Strategy pattern but is used instead to represent systems that can move between different states using well-defined actions. 
 The Singleton pattern, popular in some statically typed languages, is almost always an anti-pattern in Python.
 
## **12. Advanced Design Patterns**
In this chapter, we will be introduced to several more design patterns. Once again, we'll cover the canonical examples 
as well as any common alternative implementations in Python.  
  
### **Recall**  
Often, we'll spot really good ideas that are repeated; the repetition can form a recognizable pattern. 
Exploiting a pattern-based approach to software design can save the developer from wasting time trying to reinvent something already well understood. 
In this chapter, we looked at a few more advanced design patterns:  
* An Adapter class is a way to insert an intermediary so a client can make use of an existing class even when 
the class is not a perfect match. The software adapter parallels the idea of 
USB hardware adapters between various kinds of devices with various USB interface connectors.  
* The Façade pattern is a way to create a unified interface over a number of objects. The idea parallels the façade of 
a building that unifies separate floors, rooms, and halls into a single space.  
* We can leverage the Flyweight pattern to implement a kind of lazy initialization. Instead of copying objects, 
we can design Flyweight classes that share a common pool of data, minimizing or avoiding initialization entirely.  
* When we have closely related classes of objects, the Abstract Factory pattern can be used to build a class that can emit instances that will work together.
* The Composition pattern is widely used for complex document types. It covers programming languages, natural languages, and markup languages, including XML and HTML. 
Even something like the filesystem with a hierarchy of directories and files fits this design pattern.  
* When we have a number of similar, complex classes, it seems appropriate to create a class following the Template pattern. 
We can leave gaps or openings in the template into which we can inject any unique features.  

These patterns can help a designer focus on accepted, good design practices. Each problem is, 
of course, unique, so the patterns must be adapted. It's often better to make an adaptation to a 
known pattern and avoid trying to invent something completely new.  
  
### **Summary**  
In this chapter, we went into detail on several more design patterns, covering their canonical descriptions as well as alternatives 
for implementing them in Python, which is often more flexible and versatile than traditional object-oriented languages. 
The Adapter pattern is useful for matching interfaces, while the Façade pattern is suited to simplifying them. 
Flyweight is a complicated pattern and only useful if memory optimization is required. Abstract Factories allow the runtime separation 
of implementations depending on configuration or system information. The Composite pattern is used universally for tree-like structures. 
A Template method can be helpful for breaking complex operations into steps to avoid repeating the common features.  
  
  
## **13. Testing Object-Oriented Programs**  
Skilled Python programmers agree that testing is one of the most important aspects of software development. 
Even though this chapter is placed near the end of the book, it is not an afterthought; everything we have studied so far will help us when writing tests.  
  
### **Recall**  
In this chapter, we've looked at a number of topics related to testing applications written in Python. 
These topics include the following:  
* We described the importance of unit testing and test-driven development as a way to be sure our software does what is expected.  
* We started by using the `unittest` module because it's part of the standard library and readily available. 
It seems a little wordy, but otherwise works well for confirming that our software works.
* The `pytest` tool requires a separate installation, but it seems to produce tests that are slightly simpler than those written with the `unittest` module. 
More importantly, the sophistication of the fixture concept lets us create tests for a wide variety of scenarios.  
* The `mock` module, part of the `unittest` package, lets us create mock objects to better isolate the unit of code being tested. 
By isolating each piece of code, we can narrow our focus on being sure it works and has the right interface. 
This makes it easier to combine components.  
* Code coverage is a helpful metric to ensure that our testing is adequate. Simply adhering to a numeric goal is no substitute for thinking, 
but it can help to confirm that efforts were made to be thorough and careful when creating test scenarios.  
  
We've been looking at several kinds of tests with a variety of tools:  
  
* Unit tests with the `unittest` package or the `pytest` package, often using `Mock` objects to isolate the fixture or unit being tested.
* Integration tests, also with `unittest` and `pytest`, where more complete integrated collections of components are tested.
* Static analysis can use `mypy` to examine the data types to be sure they're used properly. This is a kind of test to ensure the software is acceptable. 
There are other kinds of static tests, and tools like `flake8`, `pylint`, and `pyflakes` can be used for these additional analyses.  
  
Some research will turn up scores of additional types of tests. Each distinct type of test has a distinct objective or approach to confirming the software works. 
A performance test, for example, seeks to establish the software is fast enough and uses an acceptable number of resources.  
  
We can't emphasize enough how important testing is. Without automated tests, software can't be considered complete, or even usable. 
Starting from test cases lets us define the expected behavior in a way that's specific, measurable, achievable, results-based, and trackable: SMART.  
  
### **Summary**  
We have finally covered the most important topic in Python programming: automated testing. Test-driven development is considered a best practice. 
The standard library `unittest` module provides a great out-of-the-box solution for testing, while the `pytest` framework has some more Pythonic syntaxes. 
Mocks can be used to emulate complex classes in our tests. Code coverage gives us an estimate of how much of our code is being run by our tests, 
but it does not tell us that we have tested the right things.  
  
  
## **14. Concurrency**  
Concurrency is the art of making a computer do (or appear to do) multiple things at once. Historically, 
this meant inviting the processor to switch between different tasks many times per second. 
In modern systems, it can also literally mean doing two or more things simultaneously on separate processor cores.

Concurrency is not inherently an object-oriented topic, but Python's concurrent systems provide object-oriented interfaces, 
as we've covered throughout the book.  
  
Concurrent processes can become complicated. The basic concepts are fairly simple, but the bugs that can occur are notoriously 
difficult to track down when the sequence of state changes is unpredictable. However, for many projects, concurrency is 
the only way to get the performance we need. Imagine if a web server couldn't respond to a user's request until another 
user's request had been completed! We'll see how to implement concurrency in Python, and some common pitfalls to avoid.
   
### **Recall**   
We've looked closely at a variety of topics related to concurrent processing in Python:  
* Threads have an advantage of simplicity for many cases. This has to be balanced against the GIL interfering with compute-intensive multi-threading.  
* Multiprocessing has an advantage of making full use of all cores of a processor. This has to be balanced against interprocess communication costs. 
If shared memory is used, there is the complication of encoding and accessing the shared objects.  
* `The concurrent.futures` module defines an abstraction – the future – that can minimize the differences in application programming 
used for accessing threads or processes. This makes it easy to switch and see which approach is fastest.  
* The `async/await` features of the Python language are supported by the AsyncIO package. Because these are coroutines, 
there isn't true parallel processing; control switches among the coroutines allow a single thread to interleave 
between waiting for I/O and computing.  
* The dining philosophers benchmark can be helpful for comparing different kinds of concurrency language features and libraries. 
It's a relatively simple problem with some interesting complexities.  
* Perhaps the most important observation is the lack of a trivial one-size-fits-all solution to concurrent processing. 
It's essential to create – and measure – a variety of solutions to determine a design that makes best use of the computing hardware.  
  
### **Summary**  
This chapter ends our exploration of object-oriented programming with a topic that isn't very object-oriented. Concurrency is a difficult problem, 
and we've only scratched the surface. While the underlying OS abstractions of processes and threads do not provide an API that is remotely object-oriented, 
Python offers some really good object-oriented abstractions around them. The threading and multiprocessing packages both provide an object-oriented 
interface to the underlying mechanics. Futures are able to encapsulate a lot of the messy details into a single object. AsyncIO uses coroutine objects 
to make our code read as though it runs synchronously, while hiding ugly and complicated implementation details behind a very simple loop abstraction.  
  


