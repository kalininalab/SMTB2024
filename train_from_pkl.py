import sys
from train import train
import pickle


with open(sys.argv[1], "rb") as f:
    args = pickle.load(f)
gpu = sys.argv[1].split(".")[-2][-1]

for dataset_path, model_name, pooling, layer_num in args:
    train(dataset_path=dataset_path, model_name=model_name, pooling=pooling, layer_num=layer_num, gpu=int(gpu), random_name="roman")
