import bisect
import collections
from collections import Counter
import csv
import heapq
import json
import random
import time
import timeit

import jsonschema


from model1 import Sample, ShufflingSamplePartition, Purpose, load
from model1 import TrainingKnownSample, TestingKnownSample
from model1 import KnownSample, MD

from collections import defaultdict, Counter
from typing import List, Any, Iterable, Callable, Tuple, Iterator, TypedDict, DefaultDict, NamedTuple, cast, Protocol
from pathlib import Path
from math import hypot
import itertools
from model1 import InvalidSampleError
from model1 import KnownSample, Purpose
import random
from model1 import ShufflingSamplePartition
from pprint import pprint

ModuloDict = DefaultDict[int, List[KnownSample]]

s2 = Sample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa")
# # print(s2)
#
s2.classification = "wrong"
# # print(s2)
#
valid = {"sepal_length": "5.1", "sepal_width": "3.5",
         "petal_length": "1.4", "petal_width": "0.2", "species": "Iris-setosa"}
# rks = TrainingKnownSample.from_dict(valid)


# #
# s3 = KnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4,
#                  petal_width=0.2, species="Iris-setosa", purpose=Purpose.Testing.value)
# # print(s3)
# # print(s3.classification is None)
#
data = [
    {
        "sepal_length": i + 0.1,
        "sepal_width": i + 0.2,
        "petal_length": i + 0.3,
        "petal_width": i + 0.4,
        "species": f"sample {i}",
    }
    for i in range(10)
]

random.seed(42)
#
ssp = ShufflingSamplePartition(data)
# pprint(ssp.testing)
#
ssp = ShufflingSamplePartition(training_subset=0.67)
for row in data:
    ssp.append(row)

x = Sample(1, 2, 3, 4)
# print(x)


s1 = TrainingKnownSample(
    sepal_length=5.1, sepal_width=3.5, petal_length=1.4,
    petal_width=0.2, species="Iris-setosa",
)
# print(s1)
TrainingKnownSample(sepal_length=5.1,
                    sepal_width=3.5, petal_length=1.4,
                    petal_width=0.2, species='Iris-setosa')

s1.classification = "wrong"
# print(s1)
# print(s1.classification)


def training(s: Sample, i: int) -> bool:
    pass


training_samples = [
    TrainingKnownSample(s)
    for i, s in enumerate(samples)
    if training(s, i)
]
test_samples = [
    TestingKnownSample(s)
    for i, s in enumerate(samples)
    if not training(s, i)
]

test_samples = list(
    TestingKnownSample(s)
    for i, s in enumerate(samples)
    if not training(s, i)
)


def training_80(s: KnownSample, i: int) -> bool:
    return i % 5 != 0


def training_75(s: KnownSample, i: int) -> bool:
    return i % 4 != 0


def training_67(s: KnownSample, i: int) -> bool:
    return i % 3 != 0


TrainingList = List[TrainingKnownSample]
TestingList = List[TestingKnownSample]


def partition(
        samples: Iterable[KnownSample],
        rule: Callable[[KnownSample, int], bool]
) -> Tuple[TrainingList, TestingList]:
    training_samples = [
        TrainingKnownSample(s)
        for i, s in enumerate(samples) if rule(s, i)
    ]
    test_samples = [
        TestingKnownSample(s)
        for i, s in enumerate(samples) if not rule(s, i)
    ]
    return training_samples, test_samples


def partition_1(
        samples: Iterable[KnownSample],
        rule: Callable[[KnownSample, int], bool]
) -> Tuple[TrainingList, TestingList]:

    training: TrainingList = []
    testing: TestingList = []
    for i, s in enumerate(samples):
        training_use = rule(s, i)
        if training_use:
            training.append(TrainingKnownSample(s))
        else:
            testing.append(TestingKnownSample(s))
    return training, testing


def partition_1p(
        samples: Iterable[KnownSample],
        rule: Callable[[KnownSample, int], bool]
) -> tuple[TrainingList, TestingList]:
    pools: defaultdict[bool, list[KnownSample]] = defaultdict(list)
    partition = ((rule(s, i), s) for i, s in enumerate(samples))
    for usage_pool, sample in partition:
        pools[usage_pool].append(sample)
    training = [TrainingKnownSample(s) for s in pools[True]]
    testing = [TestingKnownSample(s) for s in pools[False]]
    return training, testing


class CSVIrisReader:
    """
    Attribute Information:
    1. sepal length in cm
    2. sepal width in cm
    3. petal length in cm
    4. petal width in cm
    5. class:
        -- Iris Setosa
        -- Iris Versicolour
        -- Iris Virginica
    """
    header = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ]

    def __init__(self, source: Path) -> None:
        self.source = source

    def data_iter(self) -> Iterator[dict[str, str]]:
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file, self.header)
            yield from reader


class CSVIrisReader2:
    """
    Attribute Information:
    1. sepal length in cm
    2. sepal width in cm
    3. petal length in cm
    4. petal width in cm
    5. class:
        -- Iris Setosa
        -- Iris Versicolour
        -- Iris Virginica
    """

    def __init__(self, source: Path) -> None:
        self.source = source

    def data_iter(self) -> Iterator[dict[str, str]]:
        with self.source.open() as source_file:
            reader = csv.reader(source_file)
            for row in reader:
                yield dict(
                    sepal_length=row[0],
                    sepal_width=row[1],
                    petal_length=row[2],
                    petal_width=row[3],
                    species=row[4]
                )


class SampleDict(TypedDict):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str


class JSONIrisReader:
    def __init__(self, source: Path) -> None:
        self.source = source

    def data_iter(self) -> Iterator[SampleDict]:
        with self.source.open() as source_file:
            sample_list = json.load(source_file)
        yield from iter(sample_list)


class NDJSONIrisReader:
    def __init__(self, source: Path) -> None:
        self.source = source

    def data_iter(self) -> Iterator[SampleDict]:
        with self.source.open() as source_file:
            for line in source_file:
                sample = json.loads(line)
                yield sample


IRIS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2019-09/hyper-schema",
    "title": "Iris Data Schema",
    "description": "Schema of Bezdek Iris data",
    "type": "object",
    "properties": {
        "sepal_length": {
            "type": "number", "description": "Sepal Length in cm"},
        "sepal_width": {
            "type": "number", "description": "Sepal Width in cm"},
        "petal_length": {
            "type": "number", "description": "Petal Length in cm"},
        "petal_width": {
            "type": "number", "description": "Petal Width in cm"},
        "species": {
            "type": "string",
            "description": "class",
            "enum": [
                "Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        },
    },
    "required": [
"sepal_length", "sepal_width", "petal_length", "petal_width"],
}


class ValidatingNDJSONIrisReader:
    def __init__(self, source: Path, schema: dict[str, Any]) -> None:
        self.source = source
        self.validator = jsonschema.Draft7Validator(schema)

    def data_iter(self) -> Iterator[SampleDict]:
        with self.source.open() as source_file:
            for line in source_file:
                sample = json.loads(line)
                if self.validator.is_valid(sample):
                    yield sample
                else:
                    print(f"Invalid: {sample}")


def partition_2(
        samples: Iterable[KnownSample],
        training_rule: Callable[[int], bool]
) -> tuple[TrainingList, TestingList]:
    rule_multiple = 60
    partitions: ModuloDict = collections.defaultdict(list)
    for s in samples:
        partitions[hash(s) % rule_multiple].append(s)
    training_partitions: list[Iterator[TrainingKnownSample]] = []
    testing_partitions: list[Iterator[TestingKnownSample]] = []
    for i, p in enumerate(partitions.values()):
        if training_rule(i):
            training_partitions.append(
                TrainingKnownSample(s) for s in p
            )
        else:
            testing_partitions.append(
                TestingKnownSample(s) for s in p
            )
    training = list(itertools.chain(*training_partitions))
    testing = list(itertools.chain(*testing_partitions))
    return training, testing


p1 = range(1, 10, 2)
p2 = range(2, 10, 2)
# print(itertools.chain(p1, p2))
# print(list(itertools.chain(p1, p2)))


class AnySample:
    pass


class DistanceFunc:
    def distance(
            self,
            s1: TrainingKnownSample,
            s2: AnySample
    ) -> float:
        ...


class Euclidean(DistanceFunc):
    def distance(
            self,
            s1: TrainingKnownSample,
            s2: AnySample
    ) -> float:
        return hypot(
            (s1.sample.sample.sepal_length - s2.sample.sepal_length)**2,
            (s1.sample.sample.sepal_width - s2.sample.sepal_width)**2,
            (s1.sample.sample.petal_length - s2.sample.petal_length)**2,
            (s1.sample.sample.petal_width - s1.sample.petal_width)**2,
        )


class Manhattan(DistanceFunc):
    def distance(
            self,
            s1: TrainingKnownSample,
            s2: AnySample
    ) -> float:
        return sum(
            [
                abs(s1.sample.sample.sepal_length - s2.sample.sepal_length),
                abs(s1.sample.sample.sepal_width - s2.sample.sepal_width),
                abs(s1.sample.sample.petal_length - s2.sample.petal_length),
                abs(s1.sample.sample.petal_width - s1.sample.petal_width),
            ]
        )


class Chebyshev(DistanceFunc):
    def distance(
            self,
            s1: TrainingKnownSample,
            s2: AnySample
    ) -> float:
        return max(
            [
                abs(s1.sample.sample.sepal_length - s2.sample.sepal_length),
                abs(s1.sample.sample.sepal_width - s2.sample.sepal_width),
                abs(s1.sample.sample.petal_length - s2.sample.petal_length),
                abs(s1.sample.sample.petal_width - s1.sample.petal_width),
            ]
        )


Classifier = Callable[[int, DistanceFunc, TrainingList, AnySample], str]


class ClassifiedKnownSample:
    pass


class Hyperparameter(NamedTuple):
    k: int
    distance: DistanceFunc
    training_data: TrainingList
    classifier: Classifier
    
    def classify(self, unknown: AnySample) -> str:
        classifier = self.classifier
        distance = self.distance
        return classifier(
            self.k, self.distance.distance, self.training_data, unknown
        )
    
    def test(self, testing: TestingList) -> float:
        classifier = self.classifier
        distance = self.distance
        test_results = (
            ClassifiedKnownSample(
                t.sample,
                classifier(
                    self.k, self.distance.distance,
                    self.training_data, t.sample
                ),
            )
            for t in testing
        )
        pass_fail = map(
            lambda t: (
                1 if t.sample.species == t.classification else 0),
            test_results
        )
        return sum(pass_fail) / len(testing)
        

class Timing(NamedTuple):
    k: int
    distance_name: str
    classifier_name: str
    quality: float
    time: float    # Milliseconds


class Measured(NamedTuple):
    distance: float
    sample: TrainingKnownSample


# def knn_1(
#         k: int, dist: DistanceFunc, training_data: TrainingList, unknown: AnySample
# ) -> str:
#     distance = sorted(
#         map(lambda t: Measured(dist(t, unknown), t), training_data)
#     )
#     k_nearest = distance[:k]
#     k_frequencies: Counter[str] = collections.Counter(
#         s.sample.sample.species for s in k_nearest
#     )
#     mode, fq = k_frequencies.most_common(1)[0]
#     return mode


def k_nn_1(
        k: int,
        dist: DistanceFunc,
        training_data: TrainingList,
        unknown: AnySample
) -> str:
    distances = sorted(
        map(lambda t: Measured(dist(t, unknown), t), training_data)
    )
    k_nearest = distances[:k]
    k_frequencies: Counter[str] = Counter(
        s.sample.sample.species for s in k_nearest
    )
    mode, fq = k_frequencies.most_common(1)[0]
    return mode


def k_nn_b(
        k: int, dist: DistanceFunc, training_data: TrainingList, unknown: AnySample
) -> str:
    k_nearest = [
        Measured(float("int"), cast(TrainingKnownSample, None))
        for _ in range(k)
    ]
    for t in training_data:
        t_dist = dist(t, unknown)
        if t_dist > k_nearest[-1].distance:
            continue
        new = Measured(t_dist, t)
        k_nearest.insert(bisect.bisect_left(k_nearest, new), new)
        k_nearest.pop(-1)
    k_frequencies: Counter[str] = collections.Counter(
        s.sample.sample.species for s in k_nearest
    )
    mode, fq = k_frequencies.most_common(1)[0]
    return mode


def k_nn_q(
        k: int, dist: DistanceFunc, training_data: TrainingList, unknown: AnySample
) -> str:
    measured_iter = (
        Measured(dist(t, unknown), t) for t in training_data
    )
    k_nearest = heapq.nsmallest(k, measured_iter)
    k_frequencies: Counter[str] = collections.Counter(
        s.sample.sample.species for s in k_nearest
    )
    mode, fq = k_frequencies.most_common(1)[0]
    return mode


def test_classifier(
        training_data: List[TrainingKnownSample],
        testing_data: List[TestingKnownSample],
        classifier: Classifier
, manhattan=None) -> None:
    h = Hyperparameter(
        k=5,
        distance_function=manhattan,
        training_data=training_data,
        classifier=classifier
    )
    start = time.perf_counter()
    q = h.test(testing_data)
    end = time.perf_counter()
    print(
        f'| {classifier.__name__:10s} '
        f'| q={q:5}/{len(testing_data):5} '
        f'| {end-start:6.3f}s |'
    )


# def main() -> None:
#     test, train = a_lot_of_data(5_000)
#     print("| algorith | test quality | time |")
#     print("|----------|--------------|------|")
#

# m = timeit.timeit(
#     "manhattan(d1, d2)",
#     """
# from model1 import Sample, KnownSample, TrainingKnownSample, TestingKnownSample
# from model import manhattan, eulclidean
# d1 = TrainingKnownSample(KnownSample(Sample(1,2,3,4), "x"))
# d2 = KnownSample(Sample(2,3,4,5), "y")
# """
# )

# databis = [
#     KnownSample(sample=Sample(1, 2, 3, 4), species="a"),
#     KnownSample(sample=Sample(2, 3, 4, 5), species="b"),
#     KnownSample(sample=Sample(3, 4, 5, 6), species="c"),
#     KnownSample(sample=Sample(4, 5, 6, 7), species="d"),
#
# ]
# manhattan = Manhattan().distance
# training_data = [TrainingKnownSample(s) for s in data]
# h = Hyperparameter(1, manhattan, training_data, k_nn_1)
# h.classify(UnknownSample(sample(2, 3, 4, 5)))


class TestCommand:
    def __init__(
        self,
        hyper_param: Hyperparameter,
        testing: TestingList,
    ) -> None:
        self.hyperparameter = hyper_param
        self.testing_samples = testing

    def test(self) -> Timing:
        start = time.perf_counter()
        recall_score = self.hyperparameter.test(self.testing_samples)
        end = time.perf_counter()
        timing = Timing(
            k=self.hyperparameter.k,
            distance_name=self.hyperparameter.distance.__class__.__name__,
            classifier_name=self.hyperparameter.classifier.__name__,
            quality=recall_score,
            time=round((end - start) * 1000.0, 3),
        )
        return timing


euclidean = Euclidean()
manhattan = Manhattan()
chebyshev = Chebyshev()


def tuning(source: Path) -> None:
    train, test = load(source)
    scenarios = [
        TestCommand(Hyperparameter(k, df, train, cl), test)
        for k in range(3, 33, 2)
        for df in (euclidean, manhattan, chebyshev)
        for cl in (k_nn_1, k_nn_b, k_nn_q)
    ]
    timings = [s.test() for s in scenarios]
    for t in timings:
        if t.quality >= 1.0:
            print(t)

