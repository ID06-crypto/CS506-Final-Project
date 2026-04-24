from model_dependencies import POS_Transformer
import torch

EMBED_DIM = 768
NUM_HEADS = 8
HIDDEN_DIM1 = 512
HIDDEN_DIM2 = 256
NUM_POS_TAGS = 4

def test_output_shape():
    model = POS_Transformer(EMBED_DIM, NUM_HEADS, HIDDEN_DIM1, HIDDEN_DIM2, NUM_POS_TAGS)
    batch_size, seq_len = 4, 10
    x = torch.randn(batch_size, seq_len, EMBED_DIM)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    out = model(x, mask)
    assert out.shape == (batch_size, NUM_POS_TAGS) 

def test_output_changes_with_mask():
    model = POS_Transformer(EMBED_DIM, NUM_HEADS, HIDDEN_DIM1, HIDDEN_DIM2, NUM_POS_TAGS)
    x = torch.randn(2, 10, EMBED_DIM)
    mask_no_padding = torch.zeros(2, 10, dtype=torch.bool)
    mask_with_padding = torch.zeros(2, 10, dtype=torch.bool)
    mask_with_padding[:, 5:] = True
    out1 = model(x, mask_no_padding)
    out2 = model(x, mask_with_padding)
    assert not torch.allclose(out1, out2)

def test_no_nan_in_output():
    model = POS_Transformer(EMBED_DIM, NUM_HEADS, HIDDEN_DIM1, HIDDEN_DIM2, NUM_POS_TAGS)
    x = torch.randn(4, 10, EMBED_DIM)
    mask = torch.zeros(4, 10, dtype=torch.bool)
    out = model(x, mask)
    assert not torch.isnan(out).any()