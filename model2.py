from model1 import Sample
from model1 import TrainingKnownSample
from model1 import InvalidSampleError
from model1 import KnownSample, Purpose

s2 = Sample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa")
# print(s2)

s2.classification = "wrong"
# print(s2)

valid = {"sepal_length": "5.1", "sepal_width": "3.5",
         "petal_length": "1.4", "petal_width": "0.2", "species": "Iris-setosa"}
# rks = TrainingKnownSample.from_dict(valid)

s3 = KnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4,
                 petal_width=0.2, species="Iris-setosa", purpose=Purpose.Testing.value)
print(s3)
print(s3.classification is None)