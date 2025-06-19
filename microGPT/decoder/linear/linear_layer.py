import jax.numpy as jnp

def linear(params, X):
    w, b = params
    return X @ w + b
