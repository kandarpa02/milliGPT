import jax.random as random
from decoder.attn.attention import *

def test1():
    key = random.PRNGKey(0)
    x, k1 = random.split(key)
    X = random.normal(x, (1, 6, 4))
    params = [random.normal(k1, (4, 4)) for _ in range(3)]
    y = dot_product_attn(params, X)
    assert y.shape == (1, 6, 4)

