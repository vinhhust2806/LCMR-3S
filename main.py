import os
import torch
import wandb
import pprint
import argparse
import callbacks
import numpy as np
import nomenclature
from utils import load_args
from loggers import WandbLogger
from torch.utils.data import DataLoader
from trainer import NotALightningTrainer
from particular_model_trainers import Trainer
from sklearn.utils.class_weight import compute_class_weight

torch.manual_seed(28)

parser = argparse.ArgumentParser(description="Do stuff.")

parser.add_argument("--config_file", type=str, required=True)
parser.add_argument("--name", type=str, default="test")
parser.add_argument("--group", type=str, default="default")
parser.add_argument("--notes", type=str, default="")
parser.add_argument("--mode", type=str, default="dryrun")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--accumulation_steps", type=int, default=1)
parser.add_argument("--log_every", type=int, default=5)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--fold", type=int, default=None)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--position_embeddings", type=str, default=None)
parser.add_argument("--image_embeddings_type", type=str, default=None)
parser.add_argument("--text_embeddings_type", type=str, default=None)

args = parser.parse_args()
args, cfg = load_args(args)

pprint.pprint(args.__dict__)

os.environ["WANDB_MODE"] = args.mode
os.environ["WANDB_NAME"] = args.name
os.environ["WANDB_NOTES"] = args.notes

wandb.init(project="COLING2025", group=args.group, entity="DepressionDetection")

wandb.config.update(vars(args))
wandb.config.update({"config": cfg})

NUM_WORKERS = 1

# dataset and model
dataset = nomenclature.DATASETS[args.dataset]
model = nomenclature.MODELS[args.model]

train_dataset = dataset(args=args, kind="train")
val_dataset = dataset(args=args, kind="valid")

labels = train_dataset.labels
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y= labels)

model = model(args)
trainer = Trainer(
    args, model, class_weights=class_weights if args.use_class_weights else None
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, num_workers=NUM_WORKERS, pin_memory=True
)

wandb_logger = WandbLogger()

checkpoint_callback = callbacks.ModelCheckpoint(
    monitor="val_f1",
    direction="up",
    dirpath=f"checkpoints/{args.group}-{args.name}",
    save_weights_only=True,
    filename="epoch={epoch}-val_f1={val_f1:.6f}.ckpt",
)

lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer=trainer.configure_optimizers(lr=args.base_lr),
    cycle_momentum=False,
    base_lr=args.base_lr,
    mode="triangular",
    step_size_up=10 * len(train_dataloader) / args.accumulation_steps,  # per epoch
    max_lr=args.base_lr * 10,
)

lr_callback = callbacks.LambdaCallback(on_batch_end=lambda: lr_scheduler.step())

lr_logger = callbacks.LambdaCallback(
    on_batch_end=lambda: wandb_logger.log("lr", lr_scheduler.get_last_lr()[0])
)

acumen_trainer = NotALightningTrainer(
    args=args,
    callbacks=[checkpoint_callback, lr_callback, lr_logger],
    logger=wandb_logger,
)

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    acumen_trainer.fit(trainer, train_dataloader, val_dataloader)
