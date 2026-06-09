import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from huggingface_hub import hf_hub_download
import tiktoken
from tqdm.auto import tqdm

# Import our modular model
from model import AmadeusZeroTiny, Config

# 1. Dataset definition
class StoryDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_token_id = 50256

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, allowed_special="all")

        max_len = self.block_size + 1

        if len(tokens) < max_len:
            tokens.extend([self.pad_token_id] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        return x, y

@torch.no_grad()
def estimate_loss(model, dataloader, device, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters, device=device)
    for k, (x, y) in enumerate(dataloader):
        if k >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        # BFloat16 context for evaluation
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, loss = model(x, targets=y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

def main():
    print("Starting environment setup for Local RTX 3060 Training...")

    # Setup Device (Force CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Tokenizer (Using tiktoken GPT-2 encoding, same as your Kaggle notebook)
    print("Loading tiktoken 'gpt2' tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")

    # 3. Download the Karpathy TinyStories Parquet file
    print("Fetching 'karpathy/tinystories-gpt4-clean' from HuggingFace...")
    repo_id = "karpathy/tinystories-gpt4-clean"
    filename = "tinystories_gpt4_clean.parquet"

    try:
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        df = pd.read_parquet(downloaded_path)
        text_column = df.columns[0]
        stories = df[text_column].tolist()
        print(f"Successfully loaded {len(stories):,} stories.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Configuration for our tiny 26M parameter model
    conf = Config()

    # 4. Setup Dataloaders
    # For local testing, let's grab a chunk of the dataset
    # We use a larger batch size (16) since the model is small and fits in 6GB VRAM
    batch_size = 16

    train_dataset = StoryDataset(stories[:100000], tokenizer, conf.block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    val_dataset = StoryDataset(stories[2600000:2605000], tokenizer, conf.block_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 5. Initialize Model
    model = AmadeusZeroTiny(conf).to(device)
    print(f"Model instantiated. Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 6. Setup Optimizer
    epochs = 1
    learning_rate = 3e-4
    total_steps = len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print(f"Total steps per epoch: {total_steps:,}")
    print("Starting training loop (using BFloat16 mixed precision)...")

    # 7. Training Loop
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            # The Magic of Ampere: Native BFloat16 Autocast (Prevents NaN loss, super fast)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, loss = model(x, targets=y)

            optimizer.zero_grad(set_to_none=True)

            # Since we use bfloat16, we don't need a GradientScaler!
            loss.backward()

            # Clip gradients to prevent exploding loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Metrics
            train_loss_val = loss.item()
            train_ppl_val = math.exp(train_loss_val)

            pbar.set_postfix({"Loss": f"{train_loss_val:.4f}", "PPL": f"{train_ppl_val:.2f}"})

            # Validation Step
            if step > 0 and step % 1000 == 0:
                val_loss = estimate_loss(model, val_loader, device, eval_iters=50)
                val_perplexity = math.exp(val_loss)
                current_lr = scheduler.get_last_lr()[0]

                pbar.write(f"Validation at Step {step} -> Val Loss: {val_loss:.4f} | Val PPL: {val_perplexity:.4f} | LR: {current_lr:.6f}")

                # Save checkpoint
                print(f"Saving checkpoint at step {step}...")
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': train_loss_val,
                    'val_loss': val_loss,
                }, "amadeus_tiny_local.pt")

if __name__ == "__main__":
    main()
