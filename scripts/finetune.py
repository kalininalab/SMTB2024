import argparse

import torch

from smtb.train import train

torch.multiprocessing.set_start_method("spawn")
torch.multiprocessing.set_sharing_strategy("file_system")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--layer_num", type=int, required=True)
parser.add_argument("--pooling", type=str, default="mean")
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--num_workers", type=int, default=12)
parser.add_argument("--max_epoch", type=int, default=1000)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--early_stopping_patience", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--reduce_lr_patience", type=int, default=10)
parser.add_argument("--reduce_lr_factor", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)

config = parser.parse_args()
train(config)
