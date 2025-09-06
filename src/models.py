"""
Experimenting with implementing ViT; work adapted from:

Vision Transformer Tutorial by Philip Lippe
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html#Transformers-for-image-classification
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torchvision import models
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
from dataclasses import dataclass


@dataclass
class ResNetCfg:
    n_classes: int
    pretrained: bool
    backbone_freeze: bool
    freeze_duration: int
    learning_rate_head: float
    learning_rate_bb: float


@dataclass
class ViTConfig:
    embed_dims: int
    hidden_dims: int
    n_channels: int
    n_heads: int
    n_layers: int
    n_classes: int
    patch_size: int
    n_patches: int
    dropout: float = 0.0


def patchify(
    img: torch.Tensor,
    patch_size: int,
    flatten: bool,
):
    """Helper to break images into patches for ViT"""
    B, C, H, W = img.shape

    img = img.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)

    img = img.permute(0, 2, 4, 1, 3, 5)
    img = img.flatten(1, 2)

    if flatten:
        img = img.flatten(2, 4)
    return img  # [B, H*W, C*P*P]


class AttentionBlock(nn.Module):
    """
    This is the Norm -> Multi-Head Attention block in the Transformer Encoder
    """

    def __init__(
        self, embed_dims: int, hidden_dims: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dims)
        self.attention = nn.MultiheadAttention(embed_dims, num_heads, dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dims)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, embed_dims),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        xp = self.layer_norm_1(x)
        x = x + self.attention(xp, xp, xp)[0]
        x = x + self.ffn(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()

        self.patch_size = cfg.patch_size

        # Network layers
        self.layer_1 = nn.Linear(cfg.n_channels * (cfg.patch_size**2), cfg.embed_dims)

        self.enctrans = nn.Sequential(
            *[
                AttentionBlock(
                    cfg.embed_dims, cfg.hidden_dims, cfg.n_heads, cfg.dropout
                )
                for _ in range(cfg.n_layers)
            ]
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(cfg.embed_dims), nn.Linear(cfg.embed_dims, cfg.n_classes)
        )

        self.dropout = nn.Dropout(cfg.dropout)

        # Embeddings
        self.tokens = nn.Parameter(torch.randn(1, 1, cfg.embed_dims))
        self.positions = nn.Parameter(torch.randn(1, 1 + cfg.n_patches, cfg.embed_dims))

    def forward(self, x: torch.Tensor):
        # Break image to patches
        x = patchify(x, self.patch_size, True)
        B, T, _ = x.shape

        # N = C*P*P
        # [B, N, C*P*P] -> [B, N, D]
        x = self.layer_1(x)

        # Funny embedding magic
        tokens = self.tokens.repeat(B, 1, 1)  # repeat for each batch
        x = torch.cat([tokens, x], dim=1)
        x = x + self.positions[:, : T + 1]

        # Forward
        x = self.dropout(x)
        x = x.transpose(0, 1)

        # [B, N, D] -> [B, N, D]
        x = self.enctrans(x)

        # Get predictions
        # [B, N, D] -> [B, num_classes]
        return self.mlp(x[0])


# Lightning -> Acceleration framework
class ViT(pl.LightningModule):
    def __init__(self, cfg: ViTConfig, learning_rate=0.01) -> None:
        super().__init__()
        self.cfg = cfg

        # self.save_hyperparameters() # check on this I'm not sure
        self.model = VisionTransformer(cfg)
        self.learning_rate = learning_rate

        self.train_acc = MulticlassAccuracy(num_classes=cfg.n_classes)
        self.val_acc = MulticlassAccuracy(num_classes=cfg.n_classes)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def configure_optimizers(self):
        # Set up the optimizer
        optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[0, 100, 150, 200]
        )
        return [optimizer], [scheduler]

    def _predict(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y) # I could also do a nn.CELoss layer like in ResNet
        return loss, logits, y

    # part of Lightning framework
    def training_step(self, batch):
        loss, logits, y = self._predict(batch)
        self.train_acc.update(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch):
        loss, logits, y = self._predict(batch)
        self.val_acc.update(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)


class ResNetClassifier(pl.LightningModule):
    # Some ChatGPT help on implementing a pl version so I can use alongisde ViT training

    def __init__(self, cfg: ResNetCfg) -> None:
        super().__init__()
        self.cfg = cfg

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if cfg.pretrained else None
        self.backbone = models.resnet50(weights=weights)

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, cfg.n_classes)

        self.train_acc = MulticlassAccuracy(num_classes=cfg.n_classes)
        self.val_acc = MulticlassAccuracy(num_classes=cfg.n_classes)

        if cfg.backbone_freeze:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False

        self._unfrozen = not cfg.backbone_freeze
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        return self.backbone(x)

    def _predict(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss, logits, y

    def training_step(self, batch):
        loss, logits, y = self._predict(batch)
        self.train_acc.update(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch):
        loss, logits, y = self._predict(batch)
        self.val_acc.update(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # two param groups (head/backbone) with different LRs if both are trainable
        head_params = [
            p
            for n, p in self.backbone.named_parameters()
            if n.startswith("fc.") and p.requires_grad
        ]
        bb_params = [
            p
            for n, p in self.backbone.named_parameters()
            if not n.startswith("fc.") and p.requires_grad
        ]

        param_groups = []
        if bb_params:
            param_groups.append({"params": bb_params, "lr": self.cfg.learning_rate_bb})
        if head_params:
            param_groups.append(
                {"params": head_params, "lr": self.cfg.learning_rate_head}
            )

        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)  # type: ignore
        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        # handle warmup -> unfreeze for fine-tuning
        if (
            self.cfg.freeze_duration is not None
            and self.current_epoch + 1 == self.cfg.freeze_duration
            and not self._unfrozen
        ):
            for p in self.backbone.parameters():
                p.requires_grad = True
            self._unfrozen = True
            # Rebuild optimizer with two param groups when unfreezing happens mid-fit
            self.trainer.strategy.setup_optimizers(self.trainer)  # Lightning refresh
