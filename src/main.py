from models import ViT, ViTConfig, ResNetCfg

import pytorch_lightning as pl
from trainers import train_Vit_model
from preprocessing import get_data_loaders


MAX_EPOCHS = 30
SEED = 5301
N_CLASSES = 4

BATCH_SIZE = 10
DATA_DIR = "data"

VIT_CONFIG = ViTConfig(
    embed_dims = 256,
    hidden_dims = 512,
    n_channels = 3,
    n_heads = 8,
    n_layers = 6,
    n_classes = N_CLASSES,
    patch_size = 14,
    n_patches = 16**2,
    dropout = 0.0,
)

RESNET_SCRATCH_CONFIG = ResNetCfg(
    n_classes = N_CLASSES,
    pretrained = False,
    backbone_freeze = False,
    freeze_duration = 0,
    learning_rate_head = 0.01,
    learning_rate_bb = 0.001,
)

RESNET_FINETUNED_CONFIG = ResNetCfg(
    n_classes = N_CLASSES,
    pretrained = True,
    backbone_freeze = True,
    freeze_duration = 5,
    learning_rate_head = 0.01,
    learning_rate_bb = 0.001,
)

def main():
    """
    Could've structured this a bit better but it works!
    """
    # Seed
    pl.seed_everything(SEED)

    # Build data 
    loaders = get_data_loaders(batch_size=BATCH_SIZE, data_dir=DATA_DIR)

    # 1. Train Scratch
    # model = ResNetClassifier(resnet_config_scratch)
    # model = model.load_from_checkpoint(r"checkpoints\resnet-scratch\epoch=3-val_acc=0.99.ckpt")
    # model = train_ResNet_model(train_loader=loaders["train"], val_loader=loaders["val"], model=model, max_epochs=MAX_EPOCHS)    

    # 2. Finetuned
    # model = ResNetClassifier(resnet_config_pre)
    # model = train_ResNet_model(train_loader=loaders["train"], val_loader=loaders["val"], model=model, max_epochs=MAX_EPOCHS)    

    # 3. ViT
    # model = ViT(VIT_CONFIG)
    model = ViT.load_from_checkpoint(r"checkpoints\ViT\last.ckpt", cfg=VIT_CONFIG)
    model = train_Vit_model(train_loader=loaders["train"], val_loader=loaders["val"], model = model, max_epochs=MAX_EPOCHS)

if __name__ == "__main__":
    main()
