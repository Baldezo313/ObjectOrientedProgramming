from model1 import TrainingKnownSamples, Samples, KnownSamples

# s1 = TrainingKnownSamples(
#     sample=KnownSamples(sepal_length=5.1, sepal_width=3.5,
#                         petal_length=1.4, petal_width=0.2, species="Iris-setosa")
# )

# print(s1)
# s1.classification = "wrong"

s1 = TrainingKnownSamples(sample=KnownSamples(
    sample=Samples(sepal_length=5.1, sepal_width=3.5,
                   petal_length=1.4, petal_width=0.2),
    species="Iris-setosa"
    ),
)
print(s1)

# s1.classification = "wrong"


