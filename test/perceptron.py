from src.perceptron import perceptron_learning

def test_perceptron_learning():
    (w, b) = perceptron_learning(x=[(-1, 1), (0, -1), (10, 1)], y=[1, -1, 1])

    assert w == (7, 5)
    assert b == 3

test_perceptron_learning()
print("test_perceptron_learning has run successfully")