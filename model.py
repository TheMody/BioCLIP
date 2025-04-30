

import torch
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import *

class VisionEncoder(nn.Module):
    """DINOv2 vision backbone + projection head."""

    def __init__(self, model_name: str = "vit_base_patch16_224.dino", pretrained: bool = True,
                 proj_dim: int = 256): # for higher res image e.g. 518 use vit_base_patch14_dinov2, but much slower
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        embed_dim = self.backbone.embed_dim  # 768 for ViT‑B
        self.proj = nn.Linear(embed_dim, proj_dim, bias=False)


    def forward(self, images: torch.Tensor) -> torch.Tensor:  # images [B, 3, H, W]
            feats = self.backbone.forward_features(images)        # [B, tokens, D] or [B, D]
            if feats.ndim == 3:
                feats = feats[:, 0]  # take CLS token
            z = self.proj(feats)
            return F.normalize(z, dim=-1)


class TextEncoder(nn.Module):
    """BERT‑family text backbone + projection head."""

    def __init__(self, model_name: str = "distilbert-base-uncased", proj_dim: int = 256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, proj_dim, bias=False)

    @torch.no_grad()
    def tokenize(self, texts, device):
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return {k: v.to(device) for k, v in batch.items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        feats = output.last_hidden_state[:, 0]  # CLS token
        z = self.proj(feats)
        return F.normalize(z, dim=-1)


class CLIP(nn.Module):
    """Lightweight CLIP wrapper producing image‑text similarity logits."""

    def __init__(self, proj_dim: int = embedding_dim, temperature: float = 0.07):
        super().__init__()
        self.vision = VisionEncoder(proj_dim=proj_dim)
        self.text = TextEncoder(proj_dim=proj_dim)
        # Log‑temperature parameter as in original CLIP paper
        self.logit_scale = nn.Parameter(torch.tensor(1 / temperature).log())

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, output_embeddings = False):
        img_z = self.vision(images)                       # [B, D]
        txt_z = self.text(input_ids, attention_mask)      # [B, D]
        if output_embeddings:
            return img_z, txt_z
        scale = self.logit_scale.exp()
        logits = scale * img_z @ txt_z.t()                # [B, B]
        return logits

    @staticmethod
    def clip_loss(logits: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE loss (cross‑entropy in both directions)."""
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.t(), targets)
        return (loss_i + loss_t) / 2