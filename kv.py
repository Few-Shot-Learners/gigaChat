import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np


class KVCache:
    def __init__(self, batch_size, n_layers, n_heads, seq_len, d_k, d_v, device):
        self.k_cache = torch.empty(n_layers, batch_size, n_heads, seq_len, d_k, device=device)
        self.v_cache = torch.empty(n_layers, batch_size, n_heads, seq_len, d_v, device=device)
        self.current_length = 0

    def flush(self):
        self.current_length = 0

    def add_kv(self, K, V, layer_idx):  # K, V are (b, n_heads, t, d_k/d_v)
        self.k_cache[layer_idx, :, :, self.current_length, :] = K
        self.v_cache[layer_idx, :, :, self.current_length, :] = V
        self.current_length += 1

        return self.k_cache[layer_idx, :, :, :self.current_length-1, :], self.v_cache[layer_idx, :, :, :self.current_length-1, :], self.current_length-1
