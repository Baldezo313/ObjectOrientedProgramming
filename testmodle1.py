from __future__ import annotations

from model1 import TrainingKnownSamples, Samples, KnownSamples, Hyperparameterbis, Hyperparameter, TrainingData
from model1 import CD, ED, MD, SD
# from model2 import CSVIrisReader
from pathlib import Path
from typing import Tuple, TypedDict, List
from sympy import *
from unittest.mock import Mock, sentinel, call
import pytest
from concurrent import futures


s1 = TrainingKnownSamples(
    sample=KnownSamples(sepal_length=5.1, sepal_width=3.5,
                        petal_length=1.4, petal_width=0.2, species="Iris-setosa")
)

print(s1)
s1.classification = "wrong"
from model2 import CSVIrisReader

s1 = TrainingKnownSamples(sample=KnownSamples(
    sample=Samples(sepal_length=5.1, sepal_width=3.5,
                   petal_length=1.4, petal_width=0.2),
    species="Iris-setosa"
    ),
)
print(s1)

s1.classification = "wrong"


test_data = Path.cwd().parent/"bezdekIris.data"
rdr = CSVIrisReader(test_data)
samples = list(rdr.data_iter())
print(len(samples))
print(samples[0])

ED, k_sl, k_pl, k_sw, k_pw, u_sl, u_pl, u_sw, u_pw = symbols(
    "ED, k_sl, k_pl, k_sw, k_pw, u_sl, u_pl, u_sw, u_pw"
)
ED = sqrt((k_sl - u_sl)**2 + (k_pl - u_pl)**2 + (k_sw -u_sw)**2 + (k_pw - u_pw)**2)
print(ED)
print(pretty(ED, use_unicode=False))
e = ED.subs(dict(
    k_sl=5.1, k_sw=3.5, k_pl=1.4, k_pw=0.2,
    u_sl=7.9, u_sw=3.2, u_pl=4.7, u_pw=1.4,
))
print(e.evalf(9))


class Row(TypedDict):
    species: str
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class UnknownSample:
    pass


Known_Unknown = Tuple[TrainingKnownSamples, UnknownSample]


@pytest.fixture
def known_unknown_example_15() -> Known_Unknown:
    known_row: Row = {
        "species": "Iris-setosa",
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    k = TrainingKnownSamples(**known_row)
    unknown_row = {
        "sepal_length": 7.9,
        "sepal_width": 3.2,
        "petal_length": 4.7,
        "petal_width": 1.4,
    }
    u = UnknownSample(**unknown_row)
    return k, u


def test_ed(known_unknown_example_15: Known_Unknown) -> None:
    k, u = known_unknown_example_15
    assert ED().distance(k, u) == pytest.approx(4.50111097)


def test_cd(known_unknown_example_15: Known_Unknown) -> None:
    k, u = known_unknown_example_15
    assert CD().distance(k, u) == pytest.approx(3.3)


def test_md(known_unknown_example_15: Known_Unknown) -> None:
    k, u = known_unknown_example_15
    assert MD().distance(k, u) == pytest.approx(7.6)


# Soreen Distance
SD = sum(
    [abs(k_sl - u_sl), abs(k_sw - u_sw), abs(k_pl - u_pl), abs(k_pw - u_pw)]
) / sum([k_sl + u_sl, k_sw + u_sw, k_pl + u_pl, k_pw + u_pw])
print(pretty(SD, use_unicode=False))

e = SD.subs(dict(
    k_sl=5.1, k_sw=3.5, k_pl=1.4, k_pw=0.2,
    u_sl=7.9, u_sw=3.2, u_pl=4.7, u_pw=1.4
))
print(e.evalf(9))


def test_sd(known_unknown_example_15: Known_Unknown) -> None:
    k, u = known_unknown_example_15
    assert SD().distance(k, u) == pytest.approx(0.277372263)


@pytest.fixture
def sample_data() -> list[Mock]:
    return [
        Mock(name="Samples1", species=sentinel.Species3),
        Mock(name="Samples2", species=sentinel.Species1),
        Mock(name="Samples3", species=sentinel.Species1),
        Mock(name="Samples4", species=sentinel.Species1),
        Mock(name="Samples5", species=sentinel.Species3),
    ]


@pytest.fixture
def hyperparameter(sample_data: list[Mock]) -> Hyperparameterbis:
    mocked_distance = Mock(distance=Mock(side_effect=[11, 1, 2, 3, 13]))
    mocked_training_data = Mock(training=sample_data)
    mocked_weakref = Mock(
        return_value=mocked_training_data
    )
    fixture = Hyperparameterbis(
        k=3, algorithm=mocked_distance, training=sentinel.Unused
    )
    fixture.data = mocked_weakref
    return fixture


def test_hyperparameter(sample_data: list[Mock], hyperparameter: Mock) -> None:
    s = hyperparameter.classify(sentinel.Unknown)
    assert s == sentinel.Species1
    assert hyperparameter.algorithm.distance.mock_calls == [
        call(sentinel.Unknown, sample_data[0]),
        call(sentinel.Unknown, sample_data[1]),
        call(sentinel.Unknown, sample_data[2]),
        call(sentinel.Unknown, sample_data[3]),
        call(sentinel.Unknown, sample_data[4]),
    ]


def grid_search_1() -> None:
    td = TrainingData("iris.csv")
    source_path = Path.cwd().parent / "bezdekIris.data"
    reader = CSVIrisReader(source_path)
    td.load(reader.data_iter())
    tuning_results: List[Hyperparameter] = []
    with futures.ProcessPoolExecutor(8) as workers:
        test_runs: List[futures.Future[Hyperparameter]] = []
        for k in range(1, 41, 2):
            for algo in ED(), MD(), CD(), SD():
                h = Hyperparameter(k, algo, td)
                test_runs.append(workers.submit(h.test))
        for f in futures.as_completed(test_runs):
            tuning_results.append(f.result())
    for result in tuning_results:
        print(
            f"{result.k:2d} {result.algorithm.__class__.__name__:2s}"
            f" {result.quality:.3f}"
        )






