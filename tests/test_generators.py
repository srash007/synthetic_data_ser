from generators import FunctionGenerator




generator = FunctionGenerator(
    function="linear",
    n_samples=100
)

X, y = generator.generate()

assert X.shape[0] == 100
assert len(y) == 100

print("X shape:", X.shape)
print("y shape:", y.shape)