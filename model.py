import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config():
    seq_len = 256
    batch_size = 64
    d_model = 768
    d_k = 128
    d_v = 128
    n_heads = 6
    n_layers = 12
    d_ff = 3072
    vocab_size = 50257
    learning_rate = 3e-4
    bias = False
    dropout = 0.1


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, seq_len, dropout):
        super().__init__()
        # note that pretty much always d_k = d_v = d_model/n_heads (and so d_model should always be divisible by n_heads)
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.seq_len = seq_len
        self.w_q = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.w_k = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.w_v = nn.Linear(d_model, d_v*n_heads, bias=False)
        self.w_o = nn.Linear(n_heads * d_v, d_model)
        self.register_buffer('mask', torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1), persistent=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is (b, t, d_model)
        b, t, d_model = x.shape
        Q, K, V = self.w_q(x), self.w_k(x), self.w_v(x)  # (b, t, n_heads*d_k), (b, t, n_heads*d_k), (b, t, n_heads*d_v)
        Q = Q.view(b, t, self.n_heads, self.d_k).transpose(-2, -3)  # (b, n_heads, t, d_k)
        K = K.view(b, t, self.n_heads, self.d_k).transpose(-2, -3)  # (b, n_heads, t, d_k)
        V = V.view(b, t, self.n_heads, self.d_v).transpose(-2, -3)  # (b, n_heads, t, d_v)
        masked_attention = (Q @ K.transpose(-2, -1)).masked_fill_(self.mask[:t, :t], -float('inf'))  # (b, n_heads, t, t)
        # the reason to do mask[:t, :t] instead of just self.mask is in the case that the input sequence is shorter than the sequence length; we don't want to mask asymmetrically
        intermediate = F.softmax(masked_attention / self.d_k**0.5, dim=-1)
        intermediate = self.attention_dropout(intermediate)
        scores = intermediate @ V  # (b, n_heads, t, d_v)
        y = self.w_o(scores.transpose(1, 2).contiguous().view(b, t, self.n_heads*self.d_v))  # (b, t, n_heads*d_v) @ (n_heads*d_v, d_model) = (b, t, d_model)
        y = self.output_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.output_dropout(self.fc2(self.gelu(self.fc1(x))))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.affine = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):  # (b, t, d_model)
        mean = x.mean(dim=-1, keepdims=True)  # b, t
        std = x.std(dim=-1, keepdims=True)  # b, t
        ret = (x - mean) * self.affine / (std + self.eps) + self.bias
        return ret


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff, seq_len, dropout):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(n_heads, d_model, d_k, d_v, seq_len, dropout)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff, seq_len, n_layers, vocab_size, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.wte = nn.Embedding(vocab_size, d_model, device=device)
        self.wpe = nn.Embedding(seq_len, d_model, device=device)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_k, d_v, n_heads, d_ff, seq_len, dropout) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.apply(self._init_weights)

    def forward(self, x, targets=None):
        b, t = x.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.wte(x) + self.wpe(pos)
        x = self.emb_dropout(x)
        for block in self.blocks:
            x = block(x)
        y = self.lm_head(x)  # 4, 8, 50000
        if targets is not None:
            loss = F.cross_entropy(y.transpose(-2, -1), targets)
        else:
            loss = None
        return y, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def generate(self, prefix, tokens_to_generate=1, temperature=1.0, top_k=-1, top_p=1):
        generated = prefix.copy()
        for i in range(tokens_to_generate):
            x = torch.tensor(generated[-self.seq_len:], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = self.forward(x) # 1, 1, 50000
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                values, indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.where(logits >= values[:, -1], logits, -torch.inf)
            probs = torch.softmax(logits, dim=-1)
            
            if top_p < 1:
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True) 
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                remove_indices = cumsum > top_p # [F, F, F, T, T]; a = [1, 3, 5]; a[[F, F, F, T, T]]
                probs[0][sorted_indices[remove_indices]] = -torch.inf
                probs = torch.softmax(probs, dim=-1) # torch.Size([1, 50257])
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
        return generated[len(prefix):]



