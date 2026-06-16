"""
Binary Siamese Vision Transformer Architecture Script

This module defines a Siamese Vision Transformer (ViT) architecture used for binary
classification of building damage: damaged vs. not-damaged.
Instead of early-fusion, this model processes pre and post images independently
through a shared ViT backbone, and then fuses their high-level semantic features
(CLS tokens) to make a prediction.
"""

import torch
import torch.nn as nn

# --- Reuse shared building blocks from the multi-class ViT ---
from models_vit.vit import PatchEmbedding, TransformerBlock


class SiameseChangeViT(nn.Module):
    """
    Siamese Change Vision Transformer for Damage Detection.

    Late-fusion architecture:
    1. Pre and Post images are processed independently by the SAME backbone.
    2. The resulting [CLS] tokens are extracted.
    3. The features are fused: [cls_pre, cls_post, abs(cls_pre - cls_post)]
    4. The fused representation goes through a classification head -> single logit.
    """

    def __init__(
            self, img_size=224, patch_size=16, in_channels=3,
            embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0,
            dropout=0.1, drop_path_rate=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # --- Patch embedding (in_channels=3 because we process images individually) ---
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels, patch_size=patch_size,
            embed_dim=embed_dim, img_size=img_size
        )
        num_patches = self.patch_embed.num_patches

        # --- Positional & class tokens ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # --- Shared Transformer encoder ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                dropout=dropout, drop_path_prob=dpr[i]
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # --- Binary classification head ---
        # The fused feature vector size will be 3 * embed_dim 
        # (cls_pre, cls_post, abs(cls_pre - cls_post))
        fused_dim = embed_dim * 3
        
        self.head = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 1)          # single logit → P(damaged)
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single image through the ViT backbone.
        Returns the processed [CLS] token.
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # Return the [CLS] token

    def forward(self, pre_img: torch.Tensor, post_img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Siamese Network.

        Args:
            pre_img:  (B, 3, H, W) pre-disaster image
            post_img: (B, 3, H, W) post-disaster image

        Returns:
            logits: (B, 1) raw logit — apply sigmoid for probability
        """
        # Process independently through shared backbone
        cls_pre = self.forward_features(pre_img)
        cls_post = self.forward_features(post_img)
        
        # Calculate difference feature
        cls_diff = torch.abs(cls_pre - cls_post)
        
        # Late Fusion
        fused_features = torch.cat([cls_pre, cls_post, cls_diff], dim=1)
        
        # Predict
        logits = self.head(fused_features)

        return logits
