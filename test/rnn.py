from src.bayesian import bayesian_rule_probability

def test_bayesian_rule():
    p = bayesian_rule_probability(
      pba=0.99,  # P(B|A) - the likelihood of event B occurring given that A is true
      pbna=0.01, # P(B|!A)
      pa=0.0001, # P(A) - likelihood of event A
                 # 1 in 10000
    )
    
    assert p - 0.0098 < 0.0001

test_bayesian_rule()
print("test_bayesian_rule has run successfully")