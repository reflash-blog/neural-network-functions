def bayesian_rule_probability(pba, pbna, pa):
  pna = 1 - pa # P(!A)
  return pba * pa \
         /        \
         (pba * pa + pbna * pna)
