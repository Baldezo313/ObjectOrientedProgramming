import random

from model1 import Sample, ShufflingSamplePartition, Purpose
from model1 import TrainingKnownSample, TestingKnownSample
from model1 import KnownSample

from collections import defaultdict, Counter
from typing import List, Iterable, Callable, Tuple
# from model1 import InvalidSampleError
# from model1 import KnownSample, Purpose
# import random
# from model1 import ShufflingSamplePartition
# from pprint import pprint

#
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


# training_samples = [
#     TrainingKnownSample(s)
#     for i, s in enumerate(samples)
#     if training(s, i)
# ]
# test_samples = [
#     TestingKnownSample(s)
#     for i, s in enumerate(samples)
#     if not training(s, i)
# ]
#
# test_samples = list(
#     TestingKnownSample(s)
#     for i, s in enumerate(samples)
#     if not training(s, i)
# )


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

