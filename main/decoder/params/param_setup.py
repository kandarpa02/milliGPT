import jax.numpy as jnp
import jax.random as random


def init_linear_param(in_, out_, layer_name='', key=None):
    if key is None:
        key = random.PRNGKey(0)
    k1, _ = random.split(key)
    w = random.normal(k1, (in_,out_)) * 0.01
    b = jnp.zeros(out_)
    return {f"{layer_name}_w": w, f"{layer_name}_b":b}


def init_attention_param(d_model, layer_name='', key=0):
    keys = random.split(key, 4)

    W_q = random.normal(keys[0], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    W_k = random.normal(keys[1], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    W_v = random.normal(keys[2], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    W_o = random.normal(keys[3], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))

    return {
        f"{layer_name}_W_q": W_q,
        f"{layer_name}_W_k": W_k,
        f"{layer_name}_W_v": W_v,
        f"{layer_name}_W_o": W_o,
    }


def init_layer_norm_params(feature_dim, layer_name=''):
    gamma = jnp.ones((feature_dim,))
    beta = jnp.zeros((feature_dim,))
    return {"f{layer_name}_gamma": gamma, "f{layer_name}_beta": beta}


def init_embedding_params(key, vocab_size, d_model):
    embedding_table = random.normal(key, (vocab_size, d_model)) * 0.01
    return {"embedding_table": embedding_table}

