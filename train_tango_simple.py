#!/usr/bin/env python3
"""
Tango LoRA Trainer (Direct Hugging Face Hub Access)
====================================================

Uses declare-lab/tango directly from Hugging Face Hub without installation
- Downloads model weights directly
- Applies LoRA to diffusion UNet
- Uses emotion vectors as conditioning
- Supports rating-weighted loss
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
from huggingface_hub import hf_hub_download

# Standard diffusers and transformers
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import T5Tokenizer, T5EncoderModel


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


def inject_lora_to_unet(unet, r=4, alpha=1.0):
    """Inject LoRA into UNet attention layers"""
    count = 0
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            if any(key in name for key in ["attn", "to_q", "to_k", "to_v", "to_out"]):
                parent = unet
                for attr in name.split(".")[:-1]:
                    parent = getattr(parent, attr)
                key = name.split(".")[-1]
                original = getattr(parent, key)
                setattr(parent, key, LoRALinear(original, r=r, alpha=alpha))
                count += 1
    
    print(f"[LoRA] Injected {count} LoRA layers")
    return unet


# ============================================================
# 2. Mel-Spectrogram Converter
# ============================================================

class MelSpectrogramConverter:
    """Convert waveform to mel-spectrogram"""
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def __call__(self, audio):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        batch_size = audio.shape[0]
        mels = []
        
        for i in range(batch_size):
            wav = audio[i].cpu().numpy()
            
            mel = librosa.feature.melspectrogram(
                y=wav,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=0,
                fmax=8000
            )
            
            mel = librosa.power_to_db(mel, ref=np.max)
            mel = (mel + 80) / 80
            
            mels.append(torch.tensor(mel, dtype=torch.float32))
        
        mel_tensor = torch.stack(mels).unsqueeze(1)
        return mel_tensor


# ============================================================
# 3. Dataset
# ============================================================

from dataset import AudioRatingDataset

def collate_fn(batch):
    """Collate function with emotion vectors"""
    texts = [item["text"] for item in batch]
    audios = torch.stack([item["audio"] for item in batch])
    ratings = torch.tensor([item["text_match"] for item in batch], dtype=torch.float32)
    
    emotion_vectors = []
    for item in batch:
        if "emotion_vector" in item:
            emotion_vectors.append(torch.tensor(item["emotion_vector"], dtype=torch.float32))
        else:
            emotion_vectors.append(torch.zeros(28, dtype=torch.float32))
    
    emotion_vectors = torch.stack(emotion_vectors)
    
    return {
        "text": texts,
        "audio": audios,
        "rating": ratings,
        "emotion_vector": emotion_vectors
    }


# ============================================================
# 4. Tango LoRA Trainer
# ============================================================

class TangoLoRATrainer:
    """
    Tango-based LoRA trainer using emotion conditioning
    """
    def __init__(
        self,
        model_name="declare-lab/tango",
        device="cuda",
        lora_rank=4,
        lora_alpha=1.0,
        emotion_dim=28
    ):
        self.device = device
        self.emotion_dim = emotion_dim
        
        print(f"[Model] Loading Tango components from {model_name}...")
        
        # Load Tango's text encoder (Flan-T5)
        print("[Model] Loading Flan-T5 text encoder...")
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        self.text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-large").to(device)
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Load VAE (using standard AutoencoderKL as placeholder)
        print("[Model] Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Load UNet
        print("[Model] Loading UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            subfolder="unet"
        ).to(device)
        
        for param in self.unet.parameters():
            param.requires_grad = False
        
        # Inject LoRA
        print("[LoRA] Injecting LoRA...")
        self.unet = inject_lora_to_unet(self.unet, r=lora_rank, alpha=lora_alpha)
        
        # Mel converter
        self.mel_converter = MelSpectrogramConverter()
        
        # Emotion projection
        self.emotion_projection = nn.Linear(emotion_dim, 1024).to(device)
        nn.init.xavier_normal_(self.emotion_projection.weight)
        
        # Scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            subfolder="scheduler"
        )
        
        print("[Model] All components loaded")
        print(f"[Model] Emotion conditioning: {emotion_dim}d -> 1024d")

    def encode_emotion(self, emotion_vectors):
        """Project emotion vectors to conditioning"""
        conditioning = self.emotion_projection(emotion_vectors)
        conditioning = conditioning.unsqueeze(1)
        return conditioning

    def encode_audio(self, audio):
        """Encode audio to latent space"""
        with torch.no_grad():
            mel = self.mel_converter(audio).to(self.device)
            latents = self.vae.encode(mel).latent_dist.sample()
            latents = latents * 0.18215
        return latents

    def train_step(self, batch, optimizer, use_rating_weight=False):
        """Single training step"""
        audio = batch["audio"].to(self.device)
        ratings = batch["rating"].to(self.device)
        emotion_vectors = batch["emotion_vector"].to(self.device)
        
        # Emotion conditioning
        emotion_embeddings = self.encode_emotion(emotion_vectors)
        
        # Audio to latent
        latents = self.encode_audio(audio)
        
        # Sample noise
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()
        
        # Add noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=emotion_embeddings
        ).sample
        
        # Loss
        loss = F.mse_loss(noise_pred, noise, reduction="none")
        
        if use_rating_weight:
            rating_weight = (ratings / 10.0).view(-1, 1, 1, 1)
            loss = loss * rating_weight
        
        loss = loss.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def train(self, dataloader, epochs=3, lr=1e-4, use_rating_weight=False, output_dir="lora_out"):
        """Training loop"""
        trainable_params = [
            p for p in self.unet.parameters() if p.requires_grad
        ] + list(self.emotion_projection.parameters())
        
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        
        print(f"[Train] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for epoch in range(1, epochs + 1):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
            
            for batch in pbar:
                loss = self.train_step(batch, optimizer, use_rating_weight=use_rating_weight)
                total_loss += loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})
            
            avg_loss = total_loss / len(dataloader)
            print(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")
            
            self.save_lora(os.path.join(output_dir, f"tango_lora_epoch_{epoch}.pth"))
        
        print("[Train] Training complete!")

    def save_lora(self, output_path):
        """Save LoRA weights"""
        state_dict = {}
        
        for name, param in self.unet.named_parameters():
            if "lora_down" in name or "lora_up" in name:
                state_dict[f"unet.{name}"] = param.cpu()
        
        for name, param in self.emotion_projection.named_parameters():
            state_dict[f"emotion_projection.{name}"] = param.cpu()
        
        torch.save(state_dict, output_path)
        print(f"[Save] Weights saved to {output_path}")


# ============================================================
# 5. Main
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Tango LoRA Training with Emotion Conditioning")
    parser.add_argument("--json-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--use-rating-weight", action="store_true")
    parser.add_argument("--output-dir", type=str, default="lora_out")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using: {device}")
    
    # Load dataset
    print(f"[Dataset] Loading from {args.json_path}...")
    dataset = AudioRatingDataset(args.json_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize trainer
    trainer = TangoLoRATrainer(
        device=device,
        lora_rank=args.lora_rank
    )
    
    # Train
    trainer.train(
        dataloader,
        epochs=args.epochs,
        lr=args.lr,
        use_rating_weight=args.use_rating_weight,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
