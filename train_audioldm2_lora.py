#!/usr/bin/env python3
"""
AudioLDM2 LoRA Trainer - Final Working Version
================================================
This implementation applies LoRA to AudioLDM2's UNet attention layers
for fine-tuning on custom audio-text pairs.

Key Features:
- Automatic attention layer detection (to_q, to_k, to_v, to_out)
- Conv2d + Linear LoRA support
- Original weights frozen (LoRA only)
- AudioLDM2 native conditioning pipeline
- Gradient checkpoint disabled for LoRA compatibility
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Add AudioLDM2 to path
sys.path.insert(0, "./AudioLDM2")

try:
    from audioldm2.latent_diffusion.models.ddpm import LatentDiffusion
    from audioldm2.utils import default_audioldm_config, download_checkpoint
    from audioldm2.utilities.audio.stft import TacotronSTFT
    AUDIOLDM2_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] AudioLDM2 not available: {e}")
    AUDIOLDM2_AVAILABLE = False

from dataset import AudioRatingDataset


# ============================================================
# 1. LoRA Layer Implementations
# ============================================================

class LoRAConv2d(nn.Module):
    """
    LoRA module for Conv2d layers (q/k/v in AudioLDM2 attention)
    """
    def __init__(self, module, r=8, alpha=16):
        super().__init__()
        self.module = module
        out_c, in_c, kh, kw = module.weight.shape

        # LoRA down/up projections
        self.lora_down = nn.Conv2d(in_c, r, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(r, out_c, kernel_size=1, bias=False)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

        # Scaling factor
        self.scaling = alpha / r

        # Freeze original weights (critical for LoRA)
        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False

    def forward(self, x):
        return self.module(x) + self.lora_up(self.lora_down(x)) * self.scaling


class LoRALinear(nn.Module):
    """
    LoRA module for Linear layers (to_out in AudioLDM2 attention)
    """
    def __init__(self, module, r=8, alpha=16):
        super().__init__()
        self.module = module

        # LoRA down/up projections
        self.lora_down = nn.Linear(module.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, module.out_features, bias=False)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

        # Scaling factor
        self.scaling = alpha / r

        # Freeze original weights (critical for LoRA)
        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False

    def forward(self, x):
        return self.module(x) + self.lora_up(self.lora_down(x)) * self.scaling


# ============================================================
# 2. LoRA Injection into UNet
# ============================================================

def get_parent_module(root, name):
    """Get parent module from full module name"""
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent


def inject_lora(unet, r=8, alpha=16):
    """
    Automatically detect and inject LoRA into AudioLDM2 UNet attention layers
    
    Targets:
    - to_q, to_k, to_v: Can be Conv2d or Linear
    - to_out.0: Linear inside Sequential (critical - must target .0 only)
    """
    count = 0

    for name, module in unet.named_modules():
        # Conv2d-based q/k/v
        if isinstance(module, nn.Conv2d):
            if any(key in name for key in ["to_q", "to_k", "to_v"]):
                parent = get_parent_module(unet, name)
                child_name = name.split(".")[-1]
                setattr(parent, child_name, LoRAConv2d(module, r=r, alpha=alpha))
                count += 1

        # Linear-based q/k/v or to_out.0
        elif isinstance(module, nn.Linear):
            # Linear q/k/v
            if any(key in name for key in ["to_q", "to_k", "to_v"]):
                parent = get_parent_module(unet, name)
                child_name = name.split(".")[-1]
                setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha))
                count += 1

            # to_out.0 only (Sequential 내부)
            elif name.endswith("to_out.0"):
                parent = get_parent_module(unet, name)
                child_name = name.split(".")[-1]
                setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha))
                count += 1

    print(f"[LoRA] Injected {count} LoRA layers")
    return unet


# ============================================================
# 3. AudioLDM2 LoRA Trainer
# ============================================================

class AudioLDM2LoRATrainer:
    """
    Trainer class for AudioLDM2 with LoRA fine-tuning
    """
    def __init__(self, model_name="audioldm2-full", device="cuda", lora_rank=8, lora_alpha=16):
        # Load config
        cfg = default_audioldm_config(model_name)

        # Critical: Disable gradient checkpointing for LoRA compatibility
        cfg["model"]["params"]["unet_config"]["params"]["use_checkpoint"] = False
        cfg["model"]["params"]["device"] = device

        # Critical: Simplify conditioning to text-only (remove AudioMAE requirement)
        # This removes the "fname" requirement and makes it compatible with our dataset
        cfg["model"]["params"]["cond_stage_config"] = {
            "crossattn_flan_t5": {
                "cond_stage_key": "text",
                "conditioning_key": "crossattn",
                "target": "audioldm2.latent_diffusion.modules.encoders.modules.FlanT5HiddenState",
            }
        }

        print(f"[Model] Loading {model_name}...")
        self.model = LatentDiffusion(**cfg["model"]["params"]).to(device)

        # Load checkpoint
        ckpt = download_checkpoint(model_name)
        state = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(state["state_dict"], strict=False)
        print("[Model] Loaded successfully")

        # Freeze all parameters first
        for p in self.model.parameters():
            p.requires_grad = False

        # Inject LoRA into UNet
        print("[LoRA] Injecting...")
        unet = self.model.model.diffusion_model
        inject_lora(unet, r=lora_rank, alpha=lora_alpha)
        print("[LoRA] Injection complete")

        self.device = device

        # Initialize mel-spectrogram extractor
        self.stft = TacotronSTFT(
            filter_length=1024,
            hop_length=160,
            win_length=1024,
            n_mel_channels=128,
            sampling_rate=16000,
            mel_fmin=0,
            mel_fmax=8000
        ).to(device)
        
        # ================================
        # FINAL FIX: Force ALL STFT tensors to GPU
        # ================================
        def _move_all_stft_tensors_to_device(stft, device):
            for name, value in stft.__dict__.items():
                # numpy → tensor
                if isinstance(value, torch.Tensor):
                    setattr(stft, name, value.to(device))
                # list of tensors
                elif isinstance(value, (list, tuple)):
                    new_list = []
                    changed = False
                    for v in value:
                        if isinstance(v, torch.Tensor):
                            new_list.append(v.to(device))
                            changed = True
                        else:
                            new_list.append(v)
                    if changed:
                        setattr(stft, name, new_list)

        # apply to TacotronSTFT
        _move_all_stft_tensors_to_device(self.stft, self.device)

    def wav_to_mel(self, wav):
        """
        Convert waveform to mel-spectrogram
        Output: [B, 1, T, 128] for AudioLDM2
        """
        mel, _, _ = self.stft.mel_spectrogram(wav)
        # [B, 128, T] -> [B, 1, T, 128]
        mel = mel.unsqueeze(1).transpose(2, 3)
        return mel

    def encode_latent(self, mel):
        """
        Encode mel-spectrogram to latent space
        """
        enc = self.model.encode_first_stage(mel)
        return self.model.get_first_stage_encoding(enc)

    def train_step(self, batch, optimizer):
        """
        Single training step
        """
        texts = batch["text"]
        wav = batch["audio"]
        wav = wav.to(self.device, dtype=torch.float32)   # FIX

        # Prepare conditioning and latent (no grad)
        with torch.no_grad():
            # Simplified text-only conditioning
            # After config override, only text conditioning is used
            cond_key = list(self.model.cond_stage_model_metadata.keys())[0]
            
            cond = self.model.get_learned_conditioning(
                texts,  # Direct text input (not dict)
                key=cond_key,
                unconditional_cfg=False
            )
            
            # Audio to latent
            mel = self.wav_to_mel(wav)
            latent = self.encode_latent(mel)

        # Diffusion forward process
        t = torch.randint(0, self.model.num_timesteps, (latent.size(0),), device=self.device)
        noise = torch.randn_like(latent)
        noisy = self.model.q_sample(latent, t, noise=noise)

        # UNet forward (gradient enabled for LoRA)
        pred = self.model.apply_model(
            noisy,
            t,
            {"crossattn": [cond]} if not isinstance(cond, list) else {"crossattn": cond}
        )

        # Loss
        loss = F.mse_loss(pred, noise)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self, dataloader, epochs=3, lr=1e-4):
        """
        Full training loop
        """
        # Get trainable parameters (LoRA only)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr)

        print(f"[Train] Trainable parameters: {sum(p.numel() for p in params):,}")

        for epoch in range(1, epochs + 1):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
            
            for batch in pbar:
                loss = self.train_step(batch, optimizer)
                total_loss += loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})

            avg_loss = total_loss / len(dataloader)
            print(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")

    def save_lora(self, output_path="lora_audioldm2.pth"):
        """
        Save LoRA weights only
        """
        lora_state_dict = {}
        for name, param in self.model.named_parameters():
            if "lora_down" in name or "lora_up" in name:
                lora_state_dict[name] = param.cpu()

        torch.save(lora_state_dict, output_path)
        print(f"[Save] LoRA weights saved to {output_path}")
        print(f"[Save] Total LoRA parameters: {len(lora_state_dict)}")


# ============================================================
# 4. Collate Function
# ============================================================

def collate_fn(batch):
    """
    Collate function for DataLoader
    """
    texts = []
    audios = []
    ratings = []
    
    for item in batch:
        texts.append(str(item["text"]))
        audios.append(item["audio"])
        ratings.append(item["text_match"])
    
    return {
        "text": texts,
        "audio": torch.stack(audios),
        "text_match": torch.tensor(ratings, dtype=torch.float32)
    }


# ============================================================
# 5. Main Training Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AudioLDM2 LoRA Training")
    parser.add_argument("--json-path", type=str, required=True, help="Path to training JSON")
    parser.add_argument("--model-name", type=str, default="audioldm2-full", help="AudioLDM2 model name")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--output-dir", type=str, default="lora_out", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()

    if not AUDIOLDM2_AVAILABLE:
        print("[ERROR] AudioLDM2 not available. Please install it first.")
        return

    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize trainer
    trainer = AudioLDM2LoRATrainer(
        model_name=args.model_name,
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )

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
    print(f"[Dataset] Loaded {len(dataset)} samples")

    # Train
    trainer.train(dataloader, epochs=args.epochs, lr=args.lr)

    # Save
    output_path = os.path.join(args.output_dir, "lora_weights.pth")
    trainer.save_lora(output_path)

    print("[Done] Training complete!")


if __name__ == "__main__":
    main()
