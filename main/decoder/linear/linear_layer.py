import jax.numpy as jnp

def linear(X, params):
    w, b = params
    return X @ w + b
