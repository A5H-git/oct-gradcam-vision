import os
from models import ViT, ResNetClassifier

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

SEED = 5301
CHECKPOINT_PATH = "checkpoints"

def setup_trainer(name, max_epochs: int = 30, monitor: str = "val_acc", mode="max"):
    """
    Create trainers for models
    """

    logger = CSVLogger(save_dir="logs", name=name)

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_PATH, name),
        monitor=monitor,
        filename='{epoch}-{val_acc:.2f}',
        save_top_k=1,
        mode=mode,
        save_last=True
    )
    
    lr_monitor_cb = LearningRateMonitor("epoch")
    earlystop_cb = EarlyStopping(monitor=monitor, mode=mode, patience=8) # randoom number

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor_cb, earlystop_cb],
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=max_epochs,
        log_every_n_steps=10
    )
    return trainer, checkpoint_cb, logger

def train_ResNet_model(train_loader, val_loader, model: ResNetClassifier, max_epochs: int):
    mtype = "resnet-tuned" if model.cfg.pretrained else "resnet-scratch"
    trainer, checkpoint, _ = setup_trainer(mtype, max_epochs=max_epochs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_model = ResNetClassifier.load_from_checkpoint(checkpoint.best_model_path, cfg=model.cfg)
    return best_model

def train_Vit_model(train_loader, val_loader, max_epochs:int, model: ViT):
    mtype = "ViT"
    trainer, checkpoint, _ = setup_trainer(mtype, max_epochs=max_epochs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_model = ViT.load_from_checkpoint(checkpoint.best_model_path, cfg=model.cfg)
    return best_model
