import jax.numpy as jnp
import jax

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    norm = (x - mean) / (std + eps)
    return gamma * norm + beta

