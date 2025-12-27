from model import Config, TransformerModel
import torch
import tiktoken
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import argparse
from kv import KVCache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to model checkpoint")
    args = parser.parse_args()
    parser.add_argument("--mode", type=str, choices=["chat", "generate"], required=True, help="Mode: 'chat' for interactive chat, 'generate' for single prompt generation")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for text generation")

    cfg = Config()
    model = TransformerModel(
        cfg.d_model,
        cfg.d_k,
        cfg.d_v,
        cfg.n_heads,
        cfg.d_ff,
        cfg.seq_len,
        cfg.n_layers,
        cfg.vocab_size,
        cfg.dropout
    )
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    if args.mode == "chat":
        model.eval()
        print("Entering chat mode. Type 'exit' to quit.")
        kv_cache = KVCache(cfg.batch_size, cfg.n_layers, cfg.n_heads, cfg.seq_len, cfg.d_k, cfg.d_v, device)
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                break
            input_ids = torch.tensor(tiktoken.encoding_for_model("gpt2").encode(user_input), dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                output_ids = model.generate(input_ids, kv_cache=kv_cache)  # todo: define params (temp, topp, topk)
            output_text = tiktoken.encoding_for_model("gpt2").decode(output_ids[0].cpu().numpy())
            print(f"Bot: {output_text}")
    elif args.mode == "generate":
        model.eval()
        kv_cache = KVCache(1, cfg.n_layers, cfg.n_heads, cfg.seq_len, cfg.d_k, cfg.d_v, device)
        input_ids = torch.tensor(tiktoken.encoding_for_model("gpt2").encode(args.prompt), dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, kv_cache=kv_cache)
        output_text = tiktoken.encoding_for_model("gpt2").decode(output_ids[0].cpu().numpy())
        print(f"Generated Text: {output_text}")
