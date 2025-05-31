import jax.numpy as jnp
import jax
import numpy as np

def top_k_sampling(logits, k=20, temperature=1.0):
    """
    Samples a token from the top-k logits distribution using temperature scaling.

    Args:
        logits (jnp.ndarray): Logits array of shape (1, vocab_size).
        k (int): Number of top tokens to consider for sampling.
        temperature (float): Temperature for scaling the logits.

    Returns:
        int: Sampled token ID.
    """
    logits = logits / temperature
    top_k_values, top_k_indices = jax.lax.top_k(logits, k)
    probs = jax.nn.softmax(top_k_values)
    next_token = np.random.choice(np.array(top_k_indices[0]), p=np.array(probs[0]))
    return next_token

def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """
    Applies a repetition penalty to logits to discourage repeating tokens.

    Args:
        logits (jnp.ndarray): Logits array of shape (1, vocab_size).
        generated_ids (jnp.ndarray): Array of previously generated token IDs.
        penalty (float): Penalty factor to reduce scores of repeated tokens.

    Returns:
        jnp.ndarray: Modified logits with repetition penalties applied.
    """
    for token_id in generated_ids:
        logits = logits.at[0, token_id].divide(penalty)
    return logits

def generate(prompt, params, model, sp, max_new_tokens=50, temperature=1.0, top_k=20, repetition_penalty=1.2):
    """
    Generates text from a prompt using a language model with top-k sampling and repetition penalty.

    Args:
        prompt (str): Input text prompt.
        params (Any): Model parameters.
        model (Any): Model object with a `run_fn` method for inference.
        sp (Any): SentencePiece tokenizer object with `encode` and `decode`.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Temperature for sampling.
        top_k (int): Number of top logits to consider during sampling.
        repetition_penalty (float): Factor to penalize repetition.

    Returns:
        str: Generated text sequence.
    """
    model_fn = jax.jit(model.run_fn)

    input_ids = sp.encode(prompt, out_type=int)
    input_ids = jnp.array(input_ids)[None, :]

    for _ in range(max_new_tokens):
        logits = model_fn(input_ids, params)
        next_token_logits = logits[:, -1, :]

        next_token_logits = apply_repetition_penalty(next_token_logits, input_ids[0], penalty=repetition_penalty)

        next_token = top_k_sampling(next_token_logits, k=top_k, temperature=temperature)
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=-1)

    output_ids = input_ids[0].tolist()
    return sp.decode(output_ids)
