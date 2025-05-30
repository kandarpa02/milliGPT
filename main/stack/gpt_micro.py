from main.decoder.attn.attention import *
from main.decoder.embed.embedding import *
from main.decoder.layernorm.lnorm import *
from main.decoder.linear.linear_layer import *
from main.decoder.params.param_setup import *
import jax

class micro_gpt_1:
    def __init__(self, vocab, model_d):
        self.vocab = vocab
        self.model_d = model_d

    """ with one layer of transformer it uses about 3,840,000 parameters
    with 1000 vocab size and 384 dimension and I will try to keep the vocab
    size 1000, as my dataset Openwebtext10k is tiny, about 50 megabytes, 
    next when I add more layers I would use a much larger dataset, blend 
    of Openwebtext (0.7) and Wiki data (0.3), which is around 1 gigabyte

    so for now this one layer GPT will be here, I will run some experimental
    training to see if everything is working fine then, I will add more 
    layers to it.

    """
    @staticmethod
    def run_fn(X, params:dict):

        # Block 1
        x1 = word_embedding(params["embed"], X)
        x1 = layer_norm(params["ln1"], x1)
        x1_attn = multi_head_attention(params["attn1"], x1, 6)
        x1_attn += x1
        x1_fnn = jax.nn.gelu(linear(params["ffn1_fc"], x1_attn))
        x1_fnn = linear(params["ffn2_fc"], x1_fnn)
        x2 = x1_fnn + x1_attn


        embed_matrix = params["embed"]["embedding_table"]
        logits = x2 @ embed_matrix.T 


        return logits

    def get_params(self):
        params = {
            "embed"   : init_embedding_params(42, self.vocab, self.model_d),
            "attn1"   : init_attention_param(self.model_d, "attn1"),
            "ln1"     : init_layer_norm_params(self.model_d, "ln1"),
            "ffn1_fc" : init_linear_param(self.model_d, 2 * self.model_d, "ffn1_fc"),
            "ffn2_fc" : init_linear_param(2 * self.model_d, self.model_d, "ffn2_fc"),

        }
        return params


    def count_params(self):
        params = self.get_params()
        total = 0
        def _count(p):
            nonlocal total
            if isinstance(p, dict):
                for v in p.values():
                    _count(v)
            elif isinstance(p, jnp.ndarray):
                total += p.size

        _count(params)
        return total


