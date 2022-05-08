import csv
import datetime
import enum
import weakref
import math
from pathlib import Path
from enum import Enum
from pprint import pprint
from typing import List, Optional, Iterable, cast, Set, Iterator


class Sample:
    def __init__(
            self,
            sepal_length: float,
            sepal_width: float,
            petal_length: float,
            petal_width: float,
            species: Optional[str] = None,
    ) -> None:
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.species = species
        self.classification: Optional[str] = None

    def __repr__(self) -> str:
        if self.species is None:
            known_unknown = "UnknownSample"
        else:
            known_unknown = "KnownSample"
        if self.classification is None:
            classification = ""
        else:
            classification = f", {self.classification}"
        return (
            f"{known_unknown}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species!r}"
            f"{classification}"
            f")"
        )

    def classify(self, classification: str) -> None:
        self.classification = classification

    def matches(self) -> bool:
        return self.species == self.classification


class Hyperparameter:
    """A hyperparameter value and the overall quality of the classification"""
    def __init__(self, k: int, training: "TrainingData") -> None:
        self.k = k
        self.data: weakref.ReferenceType["TrainingData"] = weakref.ref(training)
        self.quality: float

    def test(self) -> None:
        """Run the entire test suite."""
        training_data: Optional["TrainingData"] = self.data()
        if not training_data:
            raise RuntimeError("Broken Weak Reference")
        pass_count, fail_count = 0, 0
        for sample in training_data.testing:
            sample.classification = self.classify(sample)
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1
        self.quality = pass_count / (pass_count + fail_count)

#
# class TrainingData:
#     """A set of traing Data and testing data with methods to load and test sample."""
#     def __init__(self, name: str) -> None:
#         self.name = name
#         self.uploaded: datetime.datetime
#         self.tested: datetime.datetime
#         self.training: List[Sample] = []
#         self.testing: List[Sample] = []
#         self.tuning: List[Hyperparameter] = []
#
#     def load(self,
#              raw_data_source: Iterable[dict[str, str]]) -> None:
#         """Load and partition the raw data"""
#         # for n, row in enumerate(raw_data_source):
#         #     ... filter and extract subsets
#         #     ... Create self.training and self.testing subsets
#         self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)
#
#     def test(self, parameter: Hyperparameter) -> None:
#         """Test this Hyperparameter"""
#         parameter.test()
#         self.tuning.append(parameter)
#         self.tested = datetime.datetime.now(tz=datetime.timezone.utc)
#
#     def classify(self,
#                  parameter: Hyperparameter,
#                  sample: Sample) -> Sample:
#         """Classify this Sample."""
#         classification = parameter.classify(sample)
#         sample.classify(classification)
#         return sample


class KnownSample(Sample):
    def __init__(
        self,
        species: str,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> None:
        super().__init__(
            sepal_length= sepal_length,
            sepal_width= sepal_width,
            petal_length= petal_length,
            petal_width= petal_width
        )
        self.species = species

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species!r}, "
            f")"
        )


class Distance:
    """Definition of a distance computation"""
    def distance(self, s1: Sample, s2: Sample) -> float:
        pass


class ED(Distance):    # distance euclidean
    def distance(self, s1: Sample, s2: Sample) -> float:
        return math.hypot(
            s1.sepal_length - s2.petal_length,
            s1.sepal_width - s2.sepal_width,
            s1.petal_length - s2.petal_length,
            s1.petal_width - s2.petal_width,
        )


class MD(Distance):       # distance Manhatann
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.petal_length),
                abs(s1.sepal_width - s2.petal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )


class CD(Distance):     # distance Chebyshev
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )


class SD(Distance):   # Sorensen distance
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        ) / sum(
            [
                s1.sepal_length - s2.petal_length,
                s1.sepal_width - s2.sepal_width,
                s1.petal_length - s2.petal_length,
                s1.petal_width - s2.petal_width,
            ]
        )


class InvalidSampleError(ValueError):
    """Source data file has invalid data representation"""


# @classmethod
# def from_dict(cls, row: dict[str, str]) -> "KnownSample":
#     if row["species"] not in {
#         "Iris-setosa", "Iris-versicolour", "Iris-virginica"
#     }:
#         raise InvalidSampleError(f"invalid species in {row!r}")
#     try:
#         return cls(
#             species=row["species"],
#             sepal_length=float(row["sepal_length"]),
#             sepal_width=float(row["sepal_width"]),
#             petal_length=float(row["petal_length"]),
#             petal_width=float(row["petal_width"]),
#         )
#     except ValueError as ex:
#         raise InvalidSampleError(f"invalid {row!r}")


class TrainingKnownSample(KnownSample):
    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "TrainingKnownSample":
        return cast(TrainingKnownSample, super().from_dict(row))


class TestingKnownSample:
    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "TestingKnownSample":
        return cast(TestingKnownSample, super().from_dict(row))


class OutlierError(ValueError):
    """Value lies outside the expected range."""


class Species(Enum):
    Setosa = "Iris-setosa"
    Versiocolour = "Iris-versicolour"
    Virginica = "Iris-virginica"


class Domain(Set[str]):
    def validate(self, value:str) -> str:
        if value in self:
            return value
        raise ValueError(f"invalid {value!r}")


species = Domain({"Iris-setosa", "Iris-versicolour", "Iris-virginica"})
species.validate("Iris-versicolour")
# species.validate("odobenidae")


@classmethod
def from_dict(cls, row: dict[str, str]) -> "KnownSample":
    try:
        return cls(
            species=species.validate(row["species"]),
            sepal_length=float(row["sepal_length"]),
            sepal_width=float(row["sepal_width"]),
            petal_length=float(row["petal_length"]),
            petal_width=float(row["petal_width"]),
        )
    except ValueError as ex:
        raise InvalidSampleError(f"invalid {row!r}")


class TrainingData:
    """A set of traing Data and testing data with methods to load and test sample."""
    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: list[TrainingKnownSample] = []
        self.testing: list[TestingKnownSample] = []
        self.tuning: list[Hyperparameter] = []

    def load(self,
             raw_data_source: Iterable[dict[str, str]]) -> None:
        """Load and partition the raw data"""
        for n, row in enumerate(raw_data_source):
            try:
               if n % 5 == 0:
                   test = TestingKnownSample.from_dict(row)
                   self.testing.append(test)
               else:
                   train = TrainingKnownSample.from_dict(row)
                   self.training.append(train)
            except InvalidSampleError as ex:
               print(f"Row {n+1}: {ex}")
               return

        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)

    def test(self, parameter: Hyperparameter) -> None:
        """Test this Hyperparameter"""
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(self,
                 parameter: Hyperparameter,
                 sample: Sample) -> Sample:
        """Classify this Sample."""
        classification = parameter.classify(sample)
        sample.classify(classification)
        return sample


def load(self, raw_data_iter: Iterable[dict[str, str]]) -> None:
    bad_count = 0
    for n, row in enumerate(raw_data_iter):
        try:
            if n % 5 == 0:
                test = TestingKnownSample.from_dict(row)
                self.testing.append(test)
            else:
                train = TrainingKnownSample.from_dict(row)
                self.training.append(train)
        except InvalidSampleError as ex:
            print(f"{bad_count} invalid rows")
            return
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)


class SampleReader:
    """
    See iris.names for attribute ordering in bezdekIris.data file
    """
    target_class = Sample
    header = [
        "sepal_length", "sepal_width",
        "petal_length", "petal_width", "class"
    ]

    def __init__(self, source: Path) -> None:
        self.source = source

    def sample_iter(self) -> Iterator[Sample]:
        target_class = self.target_class
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file, self.header)
            for row in reader:
                try:
                    sample = target_class(
                        sepal_length=float(row["sepal_length"]),
                        sepal_width=float(row["sepal_width"]),
                        petal_length=float(row["petal_length"]),
                        petal_width=float(row["petal_width"]),
                    )
                except ValueError as ex:
                    raise BadSampleRow(f"Invaid {row!r}") from ex
                yield sample


class BadSampleRow(ValueError):
    pass


class Purpose(enum.IntEnum):
    Classification = 0
    Testing = 1
    Training = 2


class KnownSample(Sample):
    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        purpose: int,
        species: str,
    ) -> None:
        purpose_enum = Purpose(purpose)
        if purpose_enum not in {Purpose.Training, Purpose.Testing}:
            raise ValueError(
                f"Invalid purpose: {purpose!r}: {purpose_enum}"
            )
        super().__init__(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self.purpose = purpose_enum
        self.species = species
        self._classification: Optional[str] = None

    def matches(self) -> bool:
        return self.species == self.classification

    @property
    def classification(self) -> Optional[str]:
        if self.purpose == Purpose.Testing:
            return self._classification
        else:
            raise AttributeError(f"Training samples have no classification")

    @classification.setter
    def classification(self, value: str) -> None:
        if self.purpose == Purpose.Testing:
            self._classification = value
        else:
            raise AttributeError(
                f"Training samples cannot be classified"
            )
