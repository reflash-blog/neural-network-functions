import numpy as np

def rnn_h_vector_step(htp, xt, whh, whx):
  return np.tanh(
    np.array(whh) @ np.array(htp) + 
    np.array(whx) @ np.array(xt)
  )

def rnn_yt_outputs(ht, why):
  return np.array(why) @ np.array(ht)