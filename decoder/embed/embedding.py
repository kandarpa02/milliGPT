import jax.numpy as jnp

def pos_encoding(seq_len, d_model):
    position = jnp.arange(seq_len, dtype=jnp.float32).reshape(seq_len, 1)
    i = jnp.arange(0, d_model, 2)
    _denominator = jnp.power(1000, i/d_model)
    pos_even = jnp.sin(position/_denominator)
    pos_odd = jnp.cos(position/_denominator)

    return jnp.stack([pos_even, pos_odd], axis=2)

