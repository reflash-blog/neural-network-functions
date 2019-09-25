from src.rbm import probability_neurons

def test_probability_neurons():
  b = [-2, 3]
  W = [
    [1,  2],
    [0, -1],
    [2,  1]
  ]
  v = [1, -2, 0]
  j = 0
  pr = probability_neurons(
    b[j],
    [wi[j] for wi in W],
    v
  )

  assert pr - 0.26894 < 0.00001


test_probability_neurons()
print("test_probability_neurons has run successfully")