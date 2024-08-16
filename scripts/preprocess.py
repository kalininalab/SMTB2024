import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

model_names = {
    48: "esm2_t48_15B_UR50D",
    36: "esm2_t36_3B_UR50D",
    33: "esm2_t33_650M_UR50D",
    30: "esm2_t30_150M_UR50D",
    12: "esm2_t12_35M_UR50D",
    6: "esm2_t6_8M_UR50D",
}


parser = ArgumentParser()
parser.add_argument("data_file", type=Path)
parser.add_argument("num_layers", type=int, choices=model_names.keys(), help="number of model layers")
parser.add_argument("out_dir", type=Path)
parser.add_argument("--esm_script", type=Path, default=Path("extract.py"), help="path to extract.py")
args = parser.parse_args()

out_dir = args.out_dir / "processed" / model_names[args.num_layers] / args.data_file.stem

df = pd.read_csv(args.data_file)

with open("tmp.fasta", "w") as f:
    for i, sequence in enumerate(df["primary"]):
        f.write(f">prot_{i}\n{sequence}\n")

launch_script = [
    "python",
    str(args.esm_script),
    model_names[args.num_layers],
    "tmp.fasta",
    str(out_dir),
    "--include",
    "per_tok",
    "--repr_layers",
]
launch_script.extend([str(i) for i in range(args.num_layers)])
os.system(" ".join(launch_script))
shutil.copy(args.data_file, out_dir / "df.csv")
