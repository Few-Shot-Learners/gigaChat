from model import Config, TransformerModel
import torch
import tiktoken
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_data():
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def join_records(split):
        ret = ""
        for elem in ds[split]['text']:
            ret += elem + "\n\n"
        return ret

    raw_train = join_records("train")[0:10000]
    raw_val = join_records("validation")[0:10000]

    # vocab size of 50257
    enc = tiktoken.get_encoding("gpt2")

    # encode_ordinary basically excludes special tokens
    train_ids = enc.encode(raw_train)
    val_ids = enc.encode(raw_val)
    return train_ids, val_ids


def get_data(ids, cfg):
    x = []
    y = []

    for id in range(0, len(ids) - 1 - cfg.seq_len * cfg.batch_size, cfg.batch_size * cfg.seq_len):
        x_batch = []
        y_batch = []
        for batch in range(cfg.batch_size):
            x_batch.append(ids[id+batch*cfg.seq_len:id+batch*cfg.seq_len+cfg.seq_len])
            y_batch.append(ids[id+batch*cfg.seq_len+1:id+batch*cfg.seq_len+cfg.seq_len+1])
        x.append(torch.tensor(x_batch, dtype=torch.long).to(device))
        y.append(torch.tensor(y_batch, dtype=torch.long).to(device))

    return (x, y)


def main():
    cfg = Config()
    train_ids, val_ids = preprocess_data()
    train_ids, val_ids = preprocess_data()
    train_data = get_data(train_ids, cfg)
    val_data = get_data(val_ids, cfg)
    model = TransformerModel(cfg.d_model, cfg.d_k, cfg.d_v, cfg.n_heads, cfg.d_ff, cfg.seq_len, cfg.n_layers, cfg.vocab_size)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    n_steps = 10000
    from tqdm import tqdm
    model.train()
    for step in tqdm(range(n_steps)):
        l = 0
        for batch in range(len(train_data)):
            optim.zero_grad()
            logits, loss = model(train_data[batch][0], train_data[batch][1])
            l = loss
            loss.backward()
            optim.step()
        if step % 500 == 0:
            print(f"loss: {l}")
    # eval

    model.eval()
    l = 0
    with torch.no_grad():
        for batch in range(len(val_data)):
            logits, loss = model(val_data[batch][0], val_data[batch][1])
            l += loss

    print(f"val loss: {l/len(val_data)}")


if __name__ == "__main__":
    main()