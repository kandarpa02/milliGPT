import jax
import jax.random as random
import jax.numpy as jnp


@jax.jit
def dot_product_attn(params:list, X):
    w_q, w_k, w_v = params

    Q = X @ w_q
    K = X @ w_k
    V = X @ w_v

    K_T = jnp.swapaxes(K, -1, -2)
    score = Q @ K_T

    d_k = Q.shape[-1] 
    scaled_score = score / jnp.sqrt(d_k)

    size = X.shape[1]
    mask = jnp.tril(jnp.ones((1, size, size), dtype = jnp.float32))

    scaled_score = scaled_score - 1e10 * (1 - mask)
    attn_weights = jax.nn.softmax(scaled_score, axis=-1)

    return attn_weights @ V

