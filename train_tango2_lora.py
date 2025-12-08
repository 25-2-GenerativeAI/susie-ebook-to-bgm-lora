#!/usr/bin/env python3
"""
Tango LoRA Trainer (Hugging Face Version)
==========================================

Uses declare-lab/tango from Hugging Face Hub
- Proper diffusion noise-prediction training
- LoRA applied to UNet attention
- Uses your custom dataset (text, audio_path, rating, emotion_vector)
- Supports emotion conditioning + rating-weighted loss
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from tqdm import tqdm

# Hugging Face Tango - install with: pip install git+https://github.com/declare-lab/tango.git
try:
    from tango import Tango
    TANGO_AVAILABLE = True
except ImportError:
    print("⚠️ Tango not installed. Installing now...")
    print("Run: pip install git+https://github.com/declare-lab/tango.git")
    TANGO_AVAILABLE = False

from diffusers import DDPMScheduler

# ============================================================
# 1. LoRA Module
# ============================================================

class LoRALinear(nn.Module):
    def __init__(self, module, r=4, alpha=1.0):
        super().__init__()
        self.module = module

        in_dim = module.in_features
        out_dim = module.out_features

        self.lora_down = nn.Linear(in_dim, r, bias=False)
        self.lora_up = nn.Linear(r, out_dim, bias=False)

        nn.init.normal_(self.lora_down.weight, std=0.02)
        nn.init.zeros_(self.lora_up.weight)

        self.scale = alpha / r

        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False

    def forward(self, x):
        return self.module(x) + self.scale * self.lora_up(self.lora_down(x))


def inject_lora(model, target_keywords=("to_q", "to_k", "to_v", "to_out"), r=4, alpha=1.0):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in target_keywords):
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            key = parts[-1]
            orig = getattr(parent, key)
            setattr(parent, key, LoRALinear(orig, r=r, alpha=alpha))
            count += 1

    print(f"[LoRA] Injected into {count} layers")
    return model


# ============================================================
# 2. Dataset (uses your JSON format directly)
# ============================================================

class TangoDataset(Dataset):
    def __init__(self, json_path, sample_rate=16000, duration=10):
        self.sample_rate = sample_rate
        self.target_len = sample_rate * duration
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def load_audio(self, path):
        wav, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        if len(wav) < self.target_len:
            wav = np.pad(wav, (0, self.target_len - len(wav)))
        else:
            wav = wav[:self.target_len]
        return torch.tensor(wav, dtype=torch.float32)

    def __getitem__(self, idx):
        item = self.data[idx]

        audio = self.load_audio(item["audio_path"])
        text = item["text"]
        rating = item.get("text_match", 1.0)       # your human-rating
        emo_vec = item.get("emotion_vector", None)

        if emo_vec is None:
            emo_vec = np.zeros(28)

        emo_vec = torch.tensor(emo_vec, dtype=torch.float32)

        return {
            "audio": audio,
            "text": text,
            "rating": rating,
            "emotion_vector": emo_vec
        }


def collate_fn(batch):
    audios = torch.stack([b["audio"] for b in batch])
    texts = [b["text"] for b in batch]
    ratings = torch.tensor([b["rating"] for b in batch], dtype=torch.float32)
    emotions = torch.stack([b["emotion_vector"] for b in batch])
    return {
        "audio": audios,
        "text": texts,
        "rating": ratings,
        "emotion": emotions
    }


# ============================================================
# 3. Tango2 LoRA Trainer
# ============================================================

class Tango2LoRATrainer:
    def __init__(self, device="cuda", lora_rank=4, lora_alpha=1.0):
        self.device = device

        print("[Model] Loading Tango2...")
        self.model = Tango("declare-lab/tango2")

        self.vae = self.model.autoencoder.to(device)
        self.unet = self.model.unet.to(device)
        self.tokenizer = self.model.tokenizer
        self.text_encoder = self.model.text_encoder.to(device)

        self.scheduler = DDPMScheduler.from_pretrained(
            "declare-lab/tango2",
            subfolder="scheduler"
        )

        # Freeze everything except LoRA
        for p in self.unet.parameters():
            p.requires_grad = False

        print("[LoRA] Injecting...")
        self.unet = inject_lora(self.unet, r=lora_rank, alpha=lora_alpha)

        print("[Model] Loaded successfully")

        self.proj_emo = nn.Linear(28, 768).to(device)

    def encode_text(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            text_emb = self.text_encoder(**inputs).last_hidden_state
        return text_emb

    def encode_audio(self, wav):
        wav = wav.unsqueeze(1)
        with torch.no_grad():
            z = self.vae.encode(wav).latent_dist.sample()
            z = z * 0.18215
        return z

    def train_step(self, batch, optimizer, use_rating=False):
        wav = batch["audio"].to(self.device)
        texts = batch["text"]
        ratings = batch["rating"].to(self.device)
        emo = batch["emotion"].to(self.device)

        t = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (wav.shape[0],), device=self.device
        ).long()

        text_emb = self.encode_text(texts)
        emo_emb = self.proj_emo(emo).unsqueeze(1)
        cond = torch.cat([text_emb, emo_emb], dim=1)

        z = self.encode_audio(wav)
        noise = torch.randn_like(z)
        z_noisy = self.scheduler.add_noise(z, noise, t)

        noise_pred = self.unet(z_noisy, t, encoder_hidden_states=cond).sample

        loss = (noise_pred - noise).pow(2)

        if use_rating:
            w = ratings.view(-1, 1, 1, 1) / 10.0
            loss = loss * w

        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self, dataloader, epochs=3, lr=1e-4, use_rating=False):
        trainable = [p for p in self.unet.parameters() if p.requires_grad] + \
                    list(self.proj_emo.parameters())

        optim = torch.optim.AdamW(trainable, lr=lr)

        for epoch in range(1, epochs + 1):
            total = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
            for batch in pbar:
                loss = self.train_step(batch, optim, use_rating=use_rating)
                total += loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})

            print(f"[Epoch {epoch}] avg loss = {total / len(dataloader):.4f}")


# ============================================================
# 4. MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rating", action="store_true")
    args = parser.parse_args()

    dataset = TangoDataset(args.json)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    trainer = Tango2LoRATrainer(device="cuda")
    trainer.train(loader, epochs=args.epochs, lr=args.lr, use_rating=args.rating)


if __name__ == "__main__":
    main()
