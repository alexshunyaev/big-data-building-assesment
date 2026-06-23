"""
Binary Vision Transformer Architecture Script

This module defines the Vision Transformer (ViT) architecture used for binary
classification of building damage: damaged vs. not-damaged.
The backbone is identical to the multi-class variant (CustomChangeViT) but the
classification head outputs a single logit passed through sigmoid.
"""

import torch
import torch.nn as nn

# --- Reuse shared building blocks from the multi-class ViT ---
from models_vit.vit import PatchEmbedding, TransformerBlock


class BinaryChangeViT(nn.Module):
    """
    Binary Change Vision Transformer for Damage Detection.

    Early-fusion architecture combining pre-disaster, post-disaster, and
    difference images.  Outputs a single logit → sigmoid → P(damaged).

    Architecture is identical to CustomChangeViT except:
      • num_classes is fixed at 1 (binary)
      • The head outputs a raw logit (apply sigmoid / BCEWithLogitsLoss externally)
    """

    def __init__(
            self, img_size=224, patch_size=16, in_channels=9,
            embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0,
            dropout=0.1, drop_path_rate=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # --- Patch embedding (shared impl) ---
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels, patch_size=patch_size,
            embed_dim=embed_dim, img_size=img_size
        )
        num_patches = self.patch_embed.num_patches

        # --- Positional & class tokens ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # --- Transformer encoder ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                dropout=dropout, drop_path_prob=dpr[i]
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # --- Binary classification head (single logit output) ---
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, 1)          # single logit → P(damaged)
        )

        # --- Weight initialisation ---
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initializes weights for linear layers and normalizations."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, pre_img: torch.Tensor, post_img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pre_img:  (B, 3, H, W) pre-disaster image
            post_img: (B, 3, H, W) post-disaster image

        Returns:
            logits: (B, 1) raw logit — apply sigmoid for probability
        """
        diff = torch.abs(post_img - pre_img)
        x = torch.cat([pre_img, post_img, diff], dim=1)   # (B, 9, H, W)
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)                        # (B, 1)

        return logits
