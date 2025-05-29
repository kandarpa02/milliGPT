import jax
import jax.numpy as jnp
from jax import lax

def pos_encoding(seq_len, d_model):
    i = jnp.arange(d_model)
    even_i = i[::2]
    denominator = jnp.power(10000.0, even_i / d_model)

    def body_fn(pos, _):
        position = jnp.array(pos, dtype=jnp.float32)
        angle_args = position / denominator
        row = jnp.zeros(d_model)

        row = row.at[::2].set(jnp.sin(angle_args))
        row = row.at[1::2].set(jnp.cos(angle_args))
        return pos + 1, row

    _, out = lax.scan(body_fn, 0, None, length=seq_len)
    out = out.reshape(seq_len, d_model // 2, 2)
    return out

pos_encoding = jax.jit(pos_encoding, static_argnames=('d_model', 'seq_len'))
