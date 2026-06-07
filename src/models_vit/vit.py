import torch
import torch.nn as nn


# =========================================================================
# Original Custom ViT Components (preserved for reference / standalone use)
# =========================================================================

class PatchEmbedding(nn.Module):
    """
    Slices the image into uniform patches and projects them into a linear embedding space.
    
    Why 6 Channels?
    We concatenate the 3-channel pre-disaster and 3-channel post-disaster images to create a 6-channel input.
    This is highly effective because it allows the convolution filter to "see" the exact spatial differences
    (pre vs post) simultaneously at the pixel level before flattening. It embeds temporal change directly 
    into the token representation.
    """

    def __init__(self, in_channels=6, patch_size=16, embed_dim=256, img_size=128):
        super().__init__()
        self.patch_size = patch_size

        # Calculate the number of patches (sequence length)
        self.num_patches = (img_size // patch_size) ** 2

        # A 2D Convolution with kernel_size and stride equal to patch_size efficiently 
        # slices the image into non-overlapping patches and projects them to embed_dim.
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width] -> [B, 6, 128, 128]
        x = self.proj(x)
        # x shape after proj: [Batch, embed_dim, grid_H, grid_W] -> [B, 256, 8, 8]

        # Flatten the spatial dimensions: [Batch, embed_dim, NumPatches]
        x = x.flatten(2)

        # Transpose to [Batch, NumPatches, embed_dim] to match standard Transformer input
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Computes scaled dot-product attention across multiple heads.
    This allows the model to map dependencies between different structural patches 
    (e.g., intact roof in patch A vs. sprawling debris in patch B).
    """

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor to prevent vanishing gradients in softmax

        # Combined linear projection for Query, Key, and Value
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape  # Batch, NumPatches, Embed_Dim

        # Linear projection and split into Q, K, V
        # Shape: [B, N, 3 * C] -> [B, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        # Permute to: [3, Batch, num_heads, NumPatches, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Extract Query, Key, Value
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute Attention Scores (Dot Product of Query and Key)
        # Shape: [B, num_heads, N, head_dim] @ [B, num_heads, head_dim, N] -> [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Multiply Attention Scores with Value
        # Shape: [B, num_heads, N, N] @ [B, num_heads, N, head_dim] -> [B, num_heads, N, head_dim]
        x = (attn @ v)

        # Transpose back and flatten heads
        # Shape: [B, N, num_heads, head_dim] -> [B, N, C]
        x = x.transpose(1, 2).reshape(B, N, C)

        # Final linear projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Multi-Layer Perceptron used inside the Transformer Block."""

    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # Gaussian Error Linear Unit is standard for ViT
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Stochastic Depth per sample. 
    Drops entire residual paths during training to regularize deep transformers.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # Work with different dimensions of tensors, mostly 3D for ViT
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TransformerBlock(nn.Module):
    """
    A single Transformer Block comprising:
    1. Pre-Layer Normalization
    2. Multi-Head Self Attention
    3. Residual Connection
    4. Pre-Layer Normalization
    5. MLP
    6. Residual Connection
    """

    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1, drop_path_prob=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=hidden_features, dropout=dropout)

    def forward(self, x):
        # Pre-Norm, Attention, DropPath, Residual
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # Pre-Norm, MLP, DropPath, Residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """
    Custom Vision Transformer for 4-Class Building Damage Classification.
    (Legacy single-stream architecture — kept for reference)
    
    Expected Input Shape: [Batch, 6, 128, 128]
    Outputs: [Batch, 4]
    """

    def __init__(
            self,
            img_size=128,
            patch_size=16,
            in_channels=6,
            num_classes=4,
            embed_dim=256,
            depth=6,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.1,
            drop_path_rate=0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels, patch_size=patch_size, embed_dim=embed_dim, img_size=img_size
        )
        num_patches = self.patch_embed.num_patches

        # 2. Class Token (Learnable parameter prepended to the token sequence)
        # It aggregates information from all other tokens to form a global image representation.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. Positional Encoding
        # Adds learnable spatial structure back to the tokens (since attention is permutation invariant)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                dropout=dropout, drop_path_prob=dpr[i]
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 5. Classification Head
        # Projects the final state of the CLS token to the 4 ordinal damage classes
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]

        # 1. Embed Patches -> [B, N, C]
        x = self.patch_embed(x)

        # 2. Append CLS Token -> [B, 1+N, C]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. Add Positional Encoding and Dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Pass through Transformer Blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 5. Extract CLS Token Output
        # The CLS token is the first token in the sequence (index 0)
        cls_out = x[:, 0]

        # 6. Classification Head
        logits = self.head(cls_out)

        return logits


# =========================================================================
# Siamese ViT — Pretrained Backbone with Dual-Stream Architecture
# =========================================================================

class SiameseViT(nn.Module):
    """
    Siamese Vision Transformer for 4-Class Building Damage Classification.

    Architecture:
        pre_img  [B, 3, 224, 224] ──┐
                                     ├──▶ Shared ViT Backbone ──▶ cls_pre  [B, 384]  ──┐
        post_img [B, 3, 224, 224] ──┘                                                   ├──▶ concat [B, 768] ──▶ MLP Head ──▶ logits [B, 4]
                                                                  cls_post [B, 384]  ──┘

    Why Siamese?
        1. Each stream receives a standard 3-channel RGB image, enabling use of
           ImageNet-pretrained weights for drastically better feature extraction.
        2. Weight sharing forces the backbone to learn general building features
           rather than memorizing which channel is pre vs. post.
        3. The classification head receives the concatenated CLS tokens and learns
           to detect *change* between the two representations.
    """

    def __init__(self, num_classes=4, backbone_name='vit_small_patch16_224', pretrained=True):
        super().__init__()

        # Import timm here to avoid module-level dependency issues
        import timm

        # Shared backbone — pretrained ViT, head removed (num_classes=0 returns CLS token)
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.embed_dim  # 384 for vit_small

        # Classification head: takes concatenated CLS tokens [cls_pre || cls_post]
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim, num_classes)
        )

    def freeze_backbone(self):
        """Freeze all backbone parameters (used during warmup phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters (used after warmup phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, pre_img, post_img):
        """
        Args:
            pre_img:  [B, 3, 224, 224] — pre-disaster RGB crop
            post_img: [B, 3, 224, 224] — post-disaster RGB crop

        Returns:
            logits: [B, num_classes] — damage classification scores
        """
        # Pass both through the shared backbone independently
        cls_pre = self.backbone(pre_img)    # [B, embed_dim]
        cls_post = self.backbone(post_img)  # [B, embed_dim]

        # Concatenate CLS token representations
        combined = torch.cat([cls_pre, cls_post], dim=1)  # [B, embed_dim * 2]

        # Classification
        logits = self.classifier(combined)  # [B, num_classes]
        return logits
