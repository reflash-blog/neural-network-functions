def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def dot_product(x1, x2):
    return sum(i1 * i2 for i1, i2 in zip(x1, x2))


def sum_vector(x1, x2):
    return tuple(i1 + i2 for i1, i2 in zip(x1, x2))


def scalar_product(xs, v):
    return tuple(v * x for x in xs)