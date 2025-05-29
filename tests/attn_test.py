import jax.random as random
from main.decoder.attn.attention import *

def test():
    key = random.PRNGKey(0)
    x, k1 = random.split(key)
    X = random.normal(x, (1, 6, 4))
    params = [random.normal(k1, (4, 4)) for _ in range(4)]
    y = multi_head_attention(X, params, 4)
    assert y.shape == (1, 6, 4)

