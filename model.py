

import torch
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import *
import math

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

    def __init__(self, model_name: str ="answerdotai/ModernBERT-base", proj_dim: int = 256):# "distilbert-base-uncased"
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, proj_dim, bias=False)

    @torch.no_grad()
    def tokenize(self, texts, device):
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=32)
        return {k: v.to(device) for k, v in batch.items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        feats = output.last_hidden_state[:, 0]  # CLS token
        z = self.proj(feats)
        return F.normalize(z, dim=-1)


# -----------------------------------------------------------------------------
#  Modern 1‑D ConvNeXt‑style Blocks
# -----------------------------------------------------------------------------

def channel_last(x: torch.Tensor):
    """Convert (B, C, L) ↔ (B, L, C) toggler."""
    return x.transpose(1, 2)


class DWConvBlock(nn.Module):
    """Depthwise conv + LN + GELU + PW conv with residual."""

    def __init__(self, dim: int, drop: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.ln = nn.LayerNorm(dim)  # channels-last
        self.pw = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # (B, C, L)
        res = x
        x = self.dw(x)
        x = channel_last(x)         # (B, L, C)
        x = self.ln(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.drop(x)
        x = channel_last(x)         # back to (B, C, L)
        return x + res


# -----------------------------------------------------------------------------
#  ECGEncoderNext
# -----------------------------------------------------------------------------

class ECGEncoder(nn.Module):
    """ConvNeXt‑inspired 1‑D encoder → L2‑normed projection."""

    def __init__(self, proj_dim: int = 256, width: int = 128, depths=(2, 2, 2)):
        super().__init__()

        self.patch_embed = nn.Conv1d(12, width, kernel_size=16, stride=4, padding=6)  # (B, W, ~250)

        blocks = []
        dim = width
        for d, depth in enumerate(depths):
            for _ in range(depth):
                blocks.append(DWConvBlock(dim, drop=0.1))
            # downsample except after last stage
            if d < len(depths) - 1:
                blocks.append(nn.Sequential(
                    nn.LayerNorm(dim, eps=1e-6),
                    nn.Linear(dim, dim * 2, bias=False),
                    nn.GELU(),
                    nn.Conv1d(dim * 2, dim * 2, kernel_size=2, stride=2),  # stride‑2 pool
                ))
                dim *= 2
        self.backbone = nn.Sequential(*blocks)

        self.head_norm = nn.LayerNorm(dim)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(dim, proj_dim, bias=False),
            nn.SiLU(),
        )

    def forward(self, wave: torch.Tensor):  # (B, 12, L)
        x = self.patch_embed(wave)
        x = self.backbone(x)
        x = channel_last(x)              # (B, L, C)
        x = self.head_norm(x)
        x = x.mean(dim=1)                # global avg over sequence length
        z = self.proj(x)
        return F.normalize(z, dim=-1)



class CLIP(nn.Module):
    """Lightweight CLIP wrapper producing image‑text similarity logits."""

    def __init__(self, proj_dim: int = embedding_dim, temperature: float = 0.07):
        super().__init__()
        self.vision = VisionEncoder(proj_dim=proj_dim)
        self.text = TextEncoder(proj_dim=proj_dim)
        self.ecg = ECGEncoder(proj_dim=proj_dim)
        # self.genomics = OmicsEncoder(proj_dim=proj_dim)
        # self.urine = UrineEncoder(proj_dim=proj_dim)
        # self.blood = BloodEncoder(proj_dim=proj_dim)
        # self.protein = ProteinEncoder(proj_dim=proj_dim)
        # Log‑temperature parameter as in original CLIP paper
        self.logit_scale = nn.Parameter(torch.tensor(1 / temperature).log())

    def forward(self, images: torch.Tensor = None, text = None, ecg:torch.Tensor = None, output_embeddings = False):
        #assert that always two modalities are passed
        assert (images is not None) + (text is not None) + (ecg is not None) == 2, "exactly two modalities must be passed"
        
        if images is not None:
            img_z = self.vision(images)                       # [B, D]
        if text is not None:
            input_ids= text["input_ids"]
            attention_mask = text["attention_mask"]
            txt_z = self.text(input_ids, attention_mask)      # [B, D]
        if ecg is not None:
            ecg_z = self.ecg(ecg)                            # [B, D]
        scale = self.logit_scale.exp()
        embeddings = []
        if images is not None:
            embeddings.append(img_z)
        if text is not None:
            embeddings.append(txt_z)
        if ecg is not None:
            embeddings.append(ecg_z)
        if output_embeddings:
            return embeddings[0], embeddings[1]
        #calculate logits
        logits = scale * embeddings[0] @ embeddings[1].t()                # [B, B]
        #logits = scale * img_z @ txt_z.t()                # [B, B]
        return logits
