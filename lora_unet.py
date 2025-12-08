"""
AudioLDM1 UNet LoRA 적용 모듈
UNet (model.model.diffusion_model)의 attention 블록에만 LoRA 주입
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


def find_lora_targets(unet):
    """
    AudioLDM1의 UNet (diffusion_model) 내부에서
    LoRA를 적용할 target module 이름을 자동으로 검색
    
    Attention 블록의 Q/K/V/Out Linear layer를 찾음
    """
    target_modules = set()

    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            # Attention 블록의 Linear layers
            if "to_q" in name or "to_k" in name or "to_v" in name or "to_out" in name:
                # 문제 3 수정: 전체 모듈 경로 등록 (name.split()이 아님)
                target_modules.add(name)

    # Fallback: 못 찾으면 기본값
    if not target_modules:
        print("[LoRA] Could not auto-detect targets, using fallback...")
        return ["to_q", "to_k", "to_v", "to_out"]

    target_modules_list = list(target_modules)
    print(f"[LoRA] Found {len(target_modules_list)} attention layers")
    return target_modules_list


def apply_lora_to_unet(unet, r=8, alpha=16, dropout=0.05):
    """
    LoRA를 UNet의 attention 블록에 적용
    
    Args:
        unet: AudioLDM1의 diffusion_model (UNet2DConditionModel)
        r: LoRA rank
        alpha: LoRA alpha scaling
        dropout: LoRA dropout rate
    
    Returns:
        unet_lora: LoRA가 적용된 UNet
    """
    
    # Target module 찾기
    target_modules = find_lora_targets(unet)
    print(f"[LoRA] Target modules found: {target_modules}")

    # LoRA 설정
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none"
    )

    # LoRA 적용
    unet_lora = get_peft_model(unet, config)

    # 파라미터 통계
    trainable = sum(p.numel() for p in unet_lora.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet_lora.parameters())

    print(f"[LoRA] Applied to UNet")
    print(f"[LoRA] Trainable params: {trainable:,} / {total:,}")
    print(f"[LoRA] Trainable ratio: {trainable/total*100:.2f}%")

    return unet_lora
