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
    with 8000 vocab size and 384 dimension and I will try to keep the vocab
    size 8000, as my dataset Openwebtext10k is tiny, about 50 megabytes, 
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
            "ffn1_fc" : init_linear_param(self.model_d, 4 * self.model_d, "ffn1_fc"),
            "ffn2_fc" : init_linear_param(4 * self.model_d, self.model_d, "ffn2_fc"),

        }
        return params


    def count_params(self):
        params = self.get_params()
        total = 0

        for i in params.keys():
            if isinstance(params[i], dict):
                for j in params[i].keys():
                    total += params[i][j].size
            elif isinstance(params[i], tuple):
                for k in params[i]:
                    total += k.size

        return total

class micro_gpt_4:
    def __init__(self, vocab, model_d):
        self.vocab = vocab
        self.model_d = model_d

    """
    This version stacks four Transformer‐style blocks. Each block has:
      - LayerNorm
      - Multi-Head Attention (with 6 heads, as before)
      - Residual connection
      - Feed-Forward Network (GELU → linear)
      - Residual connection
    The final output is projected against the same embedding matrix to produce logits.
    """

    @staticmethod
    def run_fn(X, params: dict):
        x = word_embedding(params["embed"], X)            

        # ---- Block 1 ----
        x_norm = layer_norm(params["ln1"], x)
        attn_out = multi_head_attention(params["attn1"], x_norm, 6)
        attn_res = attn_out + x                             
        ffn_hidden = jax.nn.gelu(linear(params["ffn1_fc"], attn_res))
        ffn_out = linear(params["ffn1_proj"], ffn_hidden)
        x = ffn_out + attn_res                                

        # ---- Block 2 ----
        x_norm = layer_norm(params["ln2"], x)
        attn_out = multi_head_attention(params["attn2"], x_norm, 6)
        attn_res = attn_out + x
        ffn_hidden = jax.nn.gelu(linear(params["ffn2_fc"], attn_res))
        ffn_out = linear(params["ffn2_proj"], ffn_hidden)
        x = ffn_out + attn_res

        # ---- Block 3 ----
        x_norm = layer_norm(params["ln3"], x)
        attn_out = multi_head_attention(params["attn3"], x_norm, 6)
        attn_res = attn_out + x
        ffn_hidden = jax.nn.gelu(linear(params["ffn3_fc"], attn_res))
        ffn_out = linear(params["ffn3_proj"], ffn_hidden)
        x = ffn_out + attn_res

        # ---- Block 4 ----
        x_norm = layer_norm(params["ln4"], x)
        attn_out = multi_head_attention(params["attn4"], x_norm, 6)
        attn_res = attn_out + x
        ffn_hidden = jax.nn.gelu(linear(params["ffn4_fc"], attn_res))
        ffn_out = linear(params["ffn4_proj"], ffn_hidden)
        x = ffn_out + attn_res

        embed_matrix = params["embed"]["embedding_table"]   
        logits = jnp.einsum("bsm,vm->bsv", x, embed_matrix)   

        return logits

    def get_params(self):
        params = {
            "embed"      : init_embedding_params(42, self.vocab, self.model_d),

            # ---- Block 1 params ----
            "ln1"        : init_layer_norm_params(self.model_d, "ln1"),
            "attn1"      : init_attention_param(self.model_d, "attn1"),
            "ffn1_fc"    : init_linear_param(self.model_d, 4 * self.model_d, "ffn1_fc"),
            "ffn1_proj"  : init_linear_param(4 * self.model_d, self.model_d, "ffn1_proj"),

            # ---- Block 2 params ----
            "ln2"        : init_layer_norm_params(self.model_d, "ln2"),
            "attn2"      : init_attention_param(self.model_d, "attn2"),
            "ffn2_fc"    : init_linear_param(self.model_d, 4 * self.model_d, "ffn2_fc"),
            "ffn2_proj"  : init_linear_param(4 * self.model_d, self.model_d, "ffn2_proj"),

            # ---- Block 3 params ----
            "ln3"        : init_layer_norm_params(self.model_d, "ln3"),
            "attn3"      : init_attention_param(self.model_d, "attn3"),
            "ffn3_fc"    : init_linear_param(self.model_d, 4 * self.model_d, "ffn3_fc"),
            "ffn3_proj"  : init_linear_param(4 * self.model_d, self.model_d, "ffn3_proj"),

            # ---- Block 4 params ----
            "ln4"        : init_layer_norm_params(self.model_d, "ln4"),
            "attn4"      : init_attention_param(self.model_d, "attn4"),
            "ffn4_fc"    : init_linear_param(self.model_d, 4 * self.model_d, "ffn4_fc"),
            "ffn4_proj"  : init_linear_param(4 * self.model_d, self.model_d, "ffn4_proj"),
        }
        return params

    def count_params(self):
        params = self.get_params()
        total = 0

        for i in params.keys():
            if isinstance(params[i], dict):
                for j in params[i].keys():
                    total += params[i][j].size
            elif isinstance(params[i], tuple):
                for k in params[i]:
                    total += k.size

        return total


