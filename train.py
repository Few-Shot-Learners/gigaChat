from model import Config, TransformerModel
import torch
import tiktoken
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_data():
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def join_records(split):
        ret = ""
        for elem in ds[split]['text']:
            ret += elem + "\n\n"
        return ret

    raw_train = join_records("train")
    raw_val = join_records("validation")

    # vocab size of 50257
    enc = tiktoken.get_encoding("gpt2")

    # encode_ordinary basically excludes special tokens
    train_ids = enc.encode(raw_train)
    val_ids = enc.encode(raw_val)
    return train_ids, val_ids


class TokenDataset(Dataset):
    """Dataset that samples random sequences from token IDs.

    This allows:
    - Overlapping sequences to maximize data utilization
    - Random sampling for better training dynamics
    - On-the-fly batch creation (memory efficient)
    """

    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        # Maximum number of valid starting positions
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        # Get sequence starting at idx
        x = torch.tensor(self.token_ids[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.token_ids[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def main():
    cfg = Config()
    train_ids, val_ids = preprocess_data()

    # Create datasets and dataloaders
    train_dataset = TokenDataset(train_ids, cfg.seq_len)
    val_dataset = TokenDataset(val_ids, cfg.seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,  # Random sampling for better training
        num_workers=0,  # Set to 0 for simplicity, increase for performance
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # No need to shuffle validation
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    model = TransformerModel(cfg.d_model, cfg.d_k, cfg.d_v, cfg.n_heads, cfg.d_ff, cfg.seq_len, cfg.n_layers, cfg.vocab_size)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    n_epochs = 10
    from tqdm import tqdm

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optim.zero_grad()
            logits, loss = model(x_batch, y_batch)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_loss / batch_count
        print(f"Epoch {epoch+1} - Average train loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    val_batch_count = 0

    with torch.no_grad():
        for x_batch, y_batch in tqdm(val_loader, desc="Validation"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits, loss = model(x_batch, y_batch)
            total_val_loss += loss.item()
            val_batch_count += 1

    avg_val_loss = total_val_loss / val_batch_count
    print(f"Validation loss: {avg_val_loss:.4f}")


if __name__ == "__main__":
    main()
