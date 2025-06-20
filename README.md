# microGPT 
An academic implementation of GPT: only math and JAX 

---

<img src="media/6244535664590833481.jpg" alt="attn_vs_rnn_meme" width="100%">

**microGPT** is a reflection of how the original **Transformer** layers were engineered back in 2017 at **Google**. This is a **very low-level implementation** of GPT, built entirely from **math equations and JAX**.
Core components like **Self-Attention**, **Embeddings**, **LayerNorm**, and **Feedforward Networks** are implemented **from scratch**, designed to help newcomers understand the inner workings of **LLMs** — without hiding behind prebuilt abstractions.


## Setup
### Installation:
- clone the repo
- go to the project folder
- install

```bash
git clone https://github.com/kandarpa02/microGPT.git
cd microGPT
pip install .
```

### Dependencies:
- install required packages

```bash
pip install requirements.txt 
```
## User instructions
### Know your modules:
The **GPT** stacks are here [gpt_micro.py](microGPT/stack/gpt_micro.py), you will find `micro_gpt_1`, `micro_gpt_2` and `micro_gpt_4`, the previous two **micro_gpt**s were used for experimenting with smaller data such as [openwebtext10k](https://huggingface.co/datasets/stas/openwebtext-10k), those are small but show we can use such compact language models for very domain specific tasks like grocery chatbot, auto-complete for edge devices like smart-watches and many more. 

However in this project I mostly focused on `micro_gpt_4` (17M parameters), which I trained on **TPU v3-8**, with a small dataset [Openwebtext1G](https://www.kaggle.com/datasets/kandarpasarkar/openwebtext1g) (around 1GB) of the original Openwebtext dataset, which is approximately 2.22% of the original size.

**training config:**
```python
import jax.numpy as jnp
import optax

num_heads = 8
epochs = 60
batch_size = 128
precision = jnp.bfloat16

scheduler = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=5e-4,          
    warmup_steps=100,          
    decay_steps=7600, 
    end_value=1e-5         
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),    
    optax.adamw(learning_rate=scheduler, weight_decay=0.01) 


)

```

Now `micro_gpt_4` takes two arguments `vocab`, `model_d`

```python
from microGPT.stack.gpt_micro import micro_gpt_4

gpt = micro_gpt_4(vocab = 9000, model_d = 512)
print(gpt.count_params())
# 17205248

```

Get the parameters and run forward pass like this

```python
import jax

params = gpt.get_params()
forward = jax.jit(gpt.run_fn, static_argnames=['num_heads']) # compile the function
logit = forward(input_ids, params, num_heads = 8)

```

The parameter initialization is manual with `seed = 0` by default, you can find the weight initializaton functions here [param_setup.py](microGPT/decoder/params/param_setup.py)

```python
def get_params(self):
    params = {
        "embed"      : init_embedding_params(42, self.vocab, self.model_d),

        "ln1"        : init_layer_norm_params(self.model_d),
        "attn1"      : init_attention_param(self.model_d),
        "ffn1_fc"    : init_linear_param(self.model_d, 4 * self.model_d),
        "ffn1_proj"  : init_linear_param(4 * self.model_d, self.model_d),
 
        "ln2"        : init_layer_norm_params(self.model_d),
        "attn2"      : init_attention_param(self.model_d),
        "ffn2_fc"    : init_linear_param(self.model_d, 4 * self.model_d),
        "ffn2_proj"  : init_linear_param(4 * self.model_d, self.model_d),

        "ln3"        : init_layer_norm_params(self.model_d),
        "attn3"      : init_attention_param(self.model_d),
        "ffn3_fc"    : init_linear_param(self.model_d, 4 * self.model_d),
        "ffn3_proj"  : init_linear_param(4 * self.model_d, self.model_d),

        "ln4"        : init_layer_norm_params(self.model_d),
        "attn4"      : init_attention_param(self.model_d),
        "ffn4_fc"    : init_linear_param(self.model_d, 4 * self.model_d),
        "ffn4_proj"  : init_linear_param(4 * self.model_d, self.model_d),
    }
    return params
```

Finally, I trained the model for 60 epochs and got around `PPL 17.85`, you can consider training it further with more diverse datasets!

I will add an inference module soon.

---

*If you like this repo, do give it a star :)*