import jax
import jax.numpy as jnp

def split_heads(x, num_heads):
    batch, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    x = x.reshape(batch, seq_len, num_heads, head_dim)
    return jnp.transpose(x, (0, 2, 1, 3)) 

def merge_heads(x):
    batch, num_heads, seq_len, head_dim = x.shape
    x = jnp.transpose(x, (0, 2, 1, 3))
    return x.reshape(batch, seq_len, num_heads * head_dim)

def multi_head_attention(params, X, num_heads):
    """
    X:     [batch, seq_len, hidden_dim]
    w_q:   [hidden_dim, hidden_dim]
    w_k:   [hidden_dim, hidden_dim]
    w_v:   [hidden_dim, hidden_dim]
    w_o:   [hidden_dim, hidden_dim]
    """
    batch, seq_len, hidden_dim = X.shape
    head_dim = hidden_dim // num_heads

    w_q, w_k, w_v, w_o = params

    Q = X @ w_q
    K = X @ w_k
    V = X @ w_v

    Q = split_heads(Q, num_heads)
    K = split_heads(K, num_heads)
    V = split_heads(V, num_heads)

    scores = Q @ jnp.swapaxes(K, -1, -2) / jnp.sqrt(head_dim)
    mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.float32))
    scores = scores - 1e10 * (1.0 - mask)

    weights = jax.nn.softmax(scores, axis=-1)
    attended = weights @ V

    merged = merge_heads(attended)
    out = merged @ w_o
    return out

multi_head_attention = jax.jit(multi_head_attention, static_argnames=["num_heads"])