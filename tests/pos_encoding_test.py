from main.decoder.embed.embedding import *

def test():
    out = pos_encoding(10, 6)
    assert out.shape == (10, 6)