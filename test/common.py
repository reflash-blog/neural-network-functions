from src.rnn import rnn_h_vector_step, rnn_yt_outputs

def test_rnn_h_vector_step():
    ht = rnn_h_vector_step(
      htp=[1,1],
      xt =[4,4],
      whh=[[0.5, 1],
          [   1, 2]],
      whx=[[ 1, 0],
           [-1, 0]]
    )

    # [ 0.9999666  -0.76159416] expected
    assert ht[0] - 1 < 0.00001
    assert ht[1] + 0.76159 < 0.00001

def test_rnn_yt_outputs():
    yt = rnn_yt_outputs(
      ht=[1, -0.76159],
      why =[[-1, 1],
            [ 1, 2]]
    )

    # [-1.76159 -0.52318] expected
    assert yt[0] + 1.76159 < 0.00001
    assert yt[1] + 0.52318 < 0.00001

test_rnn_h_vector_step()
print("test_rnn_h_vector_step has run successfully")
test_rnn_yt_outputs()
print("test_rnn_yt_outputs has run successfully")