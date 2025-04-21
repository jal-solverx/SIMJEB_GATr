import os 
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import torch
import numpy as np
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data
from utils import encode_for_goten
from train import Trainer


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(36)

# Hydra 초기화 및 구성 로드
initialize(config_path="config", version_base="1.1")
cfg = compose(config_name="config.yaml")

trainer = Trainer(cfg, shuffle=False)

# 구성 확인
print(OmegaConf.to_yaml(cfg))


#Test the encoder and the decoder


# Let's first test a very simple graph and see how it handles it

x = torch.tensor([
        [1,2,3,   4,5,6,   7,8,9],
        [9,8,7,   6,5,4,   3,2,1],
        [10,10,10,2,3,4,   4,5,6]
    ], dtype=torch.float)

bc = torch.tensor([
        [0],
        [1],
        [0]
    ], dtype=torch.long)

edge_index = torch.tensor([
        [0, 0],
        [1, 2]
    ], dtype=torch.long)

force = torch.tensor([
    [1,2,1],
    [1,2,1],
    [1,2,1]
])
#bc = torch.cat([force, bc], dim = -1)

batch = torch.tensor([
    0, 0, 0
])

graph = Data(x=x, bc=bc, edge_index=edge_index, batch = batch)
graph = graph.to(trainer.device)

print("Original graph node features (x):\n", graph.x)
print("Original graph boundary cond (bc):\n", graph.bc)
print("Edge index:\n", graph.edge_index, "\n")

# encoded_atomic_num, encoded_edge_diff, encoded_edge_vec = encode_for_goten(graph)
# print("Encoded atomic_number:", encoded_atomic_num)
# print("Encoded edge difference:\n", encoded_edge_diff, "\n")
# print("Encoded edge vector: ", encoded_edge_vec)


# With the encoder and the decoder tested, let's now define the model and check the forward pass

trainer.make_model_components(cfg)
trainer.model.to(trainer.device)

with torch.no_grad():
    h,X = trainer.model(graph)
    print("Output scalar feature shape:", h.shape)
    print("Output steerable feature shape:", X.shape)


