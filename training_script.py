import os
import random
import string
from pathlib import Path
import pickle

from joblib import Parallel, delayed


def random_string(k: int = 5):
    return "".join(random.choices(string.ascii_letters + string.digits, k=k))


def run_script(
    dataset_path: str,
    model_name: str,
    pooling: str,
    layer_num: int,
    gpu: int,
    hidden_dim: int = 512,
    batch_size: int = 1024,
    max_epoch: int = 1000,
    dropout: float = 0.2,
    early_stopping_patience: int = 20,
    lr: float = 0.001,
    reduce_lr_patience: int = 10,
    seed: int = 42,
):
    os.system(
        f"python train.py {dataset_path} {model_name} {pooling} {layer_num} {random_string()} --hidden_dim {hidden_dim} --batch_size {batch_size} --max_epoch {max_epoch} --dropout {dropout} --early_stopping_patience {early_stopping_patience} --lr {lr} --reduce_lr_patience {reduce_lr_patience} --seed {seed} --gpu {gpu}"
    )


dataset_names = ["fluorescence", "stability"]

# model_names = {
#     48: "esm2_t48_15B_UR50D",
#     36: "esm2_t36_3B_UR50D",
#     33: "esm2_t33_650M_UR50D",
#     30: "esm2_t30_150M_UR50D",
#     12: "esm2_t12_35M_UR50D",
#     6: "esm2_t6_8M_UR50D",
# }

model_names = {
    6: "esm2_t6_8M_UR50D",
    12: "esm2_t12_35M_UR50D",
    30: "esm2_t30_150M_UR50D",
}

pooling_options = ["attention", "mean"]

gpu_options = [[], [], [], []]

count = 0

for dataset in dataset_names:
    for model in model_names.keys():
        dataset_path = Path("/scratch/data")
        dataset_path = dataset_path / dataset / "processed" / model_names[model]

        for layer_num in range(model):
            for pooling in pooling_options:
                gpu_options[count % 4].append((str(dataset_path), model_names[model], pooling, layer_num))
                count += 1
for i, options in enumerate(gpu_options):
    with open(f"args_gpu_{i}.pkl", "wb") as f:
        pickle.dump(options, f)

# Parallel(n_jobs=4)(
#     delayed(run_script)(dataset_path, model_name, pooling, layer_num, gpu)
#     for dataset_path, model_name, pooling, layer_num, gpu in all_options
# )
