#!/usr/bin/env python3
"""
AudioLDM2 Diffusion LoRA Trainer - Complete Rewrite
====================================================
This is a complete rewrite based on proper Diffusion training principles.

Key Features:
- Direct component loading (UNet, VAE, Text Encoder)
- Proper noise prediction loss (not waveform loss)
- LoRA injection into UNet attention layers
- Compatible with AudioLDM2 official structure
- Supports custom dataset with text-audio pairs
- Mel-spectrogram preprocessing for VAE
- T5 text encoder support
- Saves LoRA weights for inference

Based on Stable Diffusion training methodology.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import librosa
import soundfile as sf
import numpy as np

# Diffusers and Transformers
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler


# ============================================================
# 1. LoRA Layer Implementation
# ============================================================

class LoRALayer(nn.Module):
    """
    LoRA layer for Linear modules in UNet
    """
    def __init__(self, module, r=4, alpha=1.0):
        super().__init__()
        self.module = module
        self.r = r
        self.alpha = alpha

        # LoRA projections
        self.lora_down = nn.Linear(module.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, module.out_features, bias=False)

        # Initialize
        nn.init.zeros_(self.lora_up.weight)
        nn.init.normal_(self.lora_down.weight, std=0.02)

        # Freeze original weights
        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False

    def forward(self, x):
        return self.module(x) + self.alpha * self.lora_up(self.lora_down(x))


def inject_lora_to_unet(unet, r=4, alpha=1.0, target_modules=None):
    """
    Inject LoRA into UNet Linear layers
    
    Args:
        unet: UNet2DConditionModel
        r: LoRA rank
        alpha: LoRA scaling factor
        target_modules: List of module name patterns to target (default: attention layers)
    """
    if target_modules is None:
        target_modules = ["attn", "to_q", "to_k", "to_v", "to_out"]
    
    count = 0
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should be targeted
            if any(pattern in name for pattern in target_modules):
                parent = unet
                for attr in name.split(".")[:-1]:
                    parent = getattr(parent, attr)
                key = name.split(".")[-1]
                original = getattr(parent, key)
                setattr(parent, key, LoRALayer(original, r=r, alpha=alpha))
                count += 1
    
    print(f"[LoRA] Injected {count} LoRA layers")
    return unet


# ============================================================
# 2. Mel-Spectrogram Converter
# ============================================================

class MelSpectrogramConverter:
    """
    Convert waveform to mel-spectrogram for AudioLDM2 VAE
    """
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def __call__(self, audio):
        """
        Convert audio waveform to mel-spectrogram
        
        Args:
            audio: [B, T] waveform tensor
        
        Returns:
            mel: [B, 1, n_mels, time] mel-spectrogram tensor
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        batch_size = audio.shape[0]
        mels = []
        
        for i in range(batch_size):
            wav = audio[i].cpu().numpy()
            
            # Compute mel-spectrogram using librosa
            mel = librosa.feature.melspectrogram(
                y=wav,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=0,
                fmax=8000
            )
            
            # Convert to log scale
            mel = librosa.power_to_db(mel, ref=np.max)
            
            # Normalize to [-1, 1]
            mel = (mel + 80) / 80  # Assuming -80dB as min
            
            mels.append(torch.tensor(mel, dtype=torch.float32))
        
        # Stack and add channel dimension: [B, n_mels, time] -> [B, 1, n_mels, time]
        mel_tensor = torch.stack(mels).unsqueeze(1)
        
        return mel_tensor


# ============================================================
# 3. Dataset (use existing AudioRatingDataset)
# ============================================================

from dataset import AudioRatingDataset


def collate_fn(batch):
    """Collate function for DataLoader"""
    texts = [item["text"] for item in batch]
    audios = torch.stack([item["audio"] for item in batch])
    ratings = torch.tensor([item["text_match"] for item in batch], dtype=torch.float32)
    
    # Emotion vectors (28-dimensional)
    emotion_vectors = []
    for item in batch:
        if "emotion_vector" in item:
            emotion_vectors.append(torch.tensor(item["emotion_vector"], dtype=torch.float32))
        else:
            # Fallback: zero vector if emotion_vector not present
            emotion_vectors.append(torch.zeros(28, dtype=torch.float32))
    
    emotion_vectors = torch.stack(emotion_vectors)
    
    return {
        "text": texts,
        "audio": audios,
        "rating": ratings,
        "emotion_vector": emotion_vectors
    }


# ============================================================
# 4. AudioLDM2 Diffusion LoRA Trainer
# ============================================================

class AudioLDM2DiffusionLoRATrainer:
    """
    Proper Diffusion-based LoRA trainer for AudioLDM2
    Uses emotion vectors as conditioning instead of text encoder
    """
    def __init__(
        self,
        model_name="cvssp/audioldm2-large",
        device="cuda",
        lora_rank=4,
        lora_alpha=1.0,
        emotion_dim=28
    ):
        self.device = device
        self.emotion_dim = emotion_dim
        
        print(f"[Model] Loading AudioLDM2 components from {model_name}...")
        
        # 1. VAE
        print("[Model] Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)
        self.vae.eval()  # Freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # 2. Mel-spectrogram converter
        self.mel_converter = MelSpectrogramConverter()
        
        # 3. UNet
        print("[Model] Loading UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device)
        
        # Freeze all UNet parameters first
        for param in self.unet.parameters():
            param.requires_grad = False
        
        # Inject LoRA
        print("[LoRA] Injecting LoRA into UNet...")
        self.unet = inject_lora_to_unet(self.unet, r=lora_rank, alpha=lora_alpha)
        print("[LoRA] Injection complete")
        
        # 4. Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        
        # 5. Emotion embedding projection (28d -> UNet conditioning dimension)
        # AudioLDM2 expects cross-attention with specific dimension
        # We need to project emotion vector to match expected dimension
        self.emotion_projection = nn.Linear(emotion_dim, 768).to(device)  # 768 is typical cross-attn dim
        nn.init.xavier_normal_(self.emotion_projection.weight)
        
        print("[Model] All components loaded successfully")
        print(f"[Model] Emotion conditioning: {emotion_dim}d -> 768d")

    def encode_emotion(self, emotion_vectors):
        """
        Project emotion vectors to conditioning embeddings
        
        Args:
            emotion_vectors: [B, 28] emotion probability vectors
        
        Returns:
            conditioning: [B, 1, 768] conditioning embeddings
        """
        # Project emotion vector to conditioning dimension
        conditioning = self.emotion_projection(emotion_vectors)  # [B, 768]
        
        # Add sequence dimension for cross-attention: [B, 768] -> [B, 1, 768]
        conditioning = conditioning.unsqueeze(1)
        
        return conditioning

    def encode_audio(self, audio):
        """Encode audio to latent space"""
        with torch.no_grad():
            # Audio waveform [B, T] -> mel-spectrogram [B, 1, n_mels, time]
            mel = self.mel_converter(audio).to(self.device)
            
            # Encode mel to latent space
            latents = self.vae.encode(mel).latent_dist.sample()
            latents = latents * 0.18215  # Scaling factor
        
        return latents

    def train_step(self, batch, optimizer, use_rating_weight=False):
        """
        Single training step with proper Diffusion loss
        Uses emotion vectors as conditioning and human ratings as loss weight
        
        Args:
            batch: Dict with "text", "audio", "rating", "emotion_vector"
            optimizer: Optimizer
            use_rating_weight: Whether to weight loss by rating
        
        Returns:
            loss value
        """
        audio = batch["audio"].to(self.device)
        ratings = batch["rating"].to(self.device)
        emotion_vectors = batch["emotion_vector"].to(self.device)
        
        # 1. Emotion conditioning (replaces text encoding)
        emotion_embeddings = self.encode_emotion(emotion_vectors)
        
        # 2. Audio to latent
        latents = self.encode_audio(audio)
        
        # 3. Sample noise and timesteps
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()
        
        # 4. Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 5. Predict noise with UNet (using emotion conditioning)
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=emotion_embeddings
        ).sample
        
        # 6. Compute loss
        loss = F.mse_loss(noise_pred, noise, reduction="none")
        
        # 7. Weight by human rating (human preference learning)
        if use_rating_weight:
            # Normalize ratings to [0, 1] range if needed (assuming 0-10 scale)
            rating_weight = (ratings / 10.0).view(-1, 1, 1, 1)
            loss = loss * rating_weight
        
        loss = loss.mean()
        
        # 8. Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def train(
        self,
        dataloader,
        epochs=3,
        lr=1e-4,
        use_rating_weight=False,
        output_dir="lora_out"
    ):
        """
        Full training loop
        """
        # Get trainable parameters (LoRA + emotion projection)
        trainable_params = [
            p for p in self.unet.parameters() if p.requires_grad
        ] + list(self.emotion_projection.parameters())
        
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        
        print(f"[Train] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"[Train] - LoRA params: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad):,}")
        print(f"[Train] - Emotion projection params: {sum(p.numel() for p in self.emotion_projection.parameters()):,}")
        print(f"[Train] Rating weighting: {use_rating_weight}")
        print(f"[Train] Conditioning: Emotion vectors (28d -> 768d)")
        
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
            
            # Save checkpoint
            self.save_lora(os.path.join(output_dir, f"lora_epoch_{epoch}.pth"))
        
        print("[Train] Training complete!")

    def save_lora(self, output_path):
        """Save LoRA weights and emotion projection"""
        state_dict = {}
        
        # Save LoRA weights
        for name, param in self.unet.named_parameters():
            if "lora_down" in name or "lora_up" in name:
                state_dict[f"unet.{name}"] = param.cpu()
        
        # Save emotion projection
        for name, param in self.emotion_projection.named_parameters():
            state_dict[f"emotion_projection.{name}"] = param.cpu()
        
        torch.save(state_dict, output_path)
        print(f"[Save] Weights saved to {output_path}")
        print(f"[Save] - LoRA parameters: {sum(1 for k in state_dict.keys() if 'lora' in k)}")
        print(f"[Save] - Emotion projection parameters: {sum(1 for k in state_dict.keys() if 'emotion_projection' in k)}")


# ============================================================
# 5. Main Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AudioLDM2 Diffusion LoRA Training with Emotion Conditioning")
    parser.add_argument("--json-path", type=str, required=True, help="Path to JSON dataset (must include emotion_vector field)")
    parser.add_argument("--model-name", type=str, default="cvssp/audioldm2-large", help="AudioLDM2 model name")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=1.0, help="LoRA alpha")
    parser.add_argument("--use-rating-weight", action="store_true", help="Weight loss by human rating (text_match score)")
    parser.add_argument("--output-dir", type=str, default="lora_out", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using: {device}")
    
    # Load dataset using existing AudioRatingDataset
    print(f"[Dataset] Loading from {args.json_path}...")
    dataset = AudioRatingDataset(args.json_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    print(f"[Dataset] Loaded {len(dataset)} samples")
    
    # Initialize trainer
    trainer = AudioLDM2DiffusionLoRATrainer(
        model_name=args.model_name,
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )
    
    # Train
    trainer.train(
        dataloader,
        epochs=args.epochs,
        lr=args.lr,
        use_rating_weight=args.use_rating_weight,
        output_dir=args.output_dir
    )
    
    print("[Done] Training complete!")


if __name__ == "__main__":
    main()
