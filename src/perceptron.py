from src.common import sign, dot_product, sum_vector, scalar_product


def heaviside_step_function(v):
    return 1 if v >= 0 else 0


def perceptron_learning(x, y):
    assert len(x) == len(y)
    assert len(x) is not 0

    w = (0, ) * len(x[0])
    b = 0

    converged = False
    while not converged:
        converged = True
        for i in range(len(x)):
            if sign(dot_product(w, x[i]) + b) != sign(y[i]):
                converged = False
                w = sum_vector(w, scalar_product(x[i], y[i]))
                b += y[i]

    return (w, b)
