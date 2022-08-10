import ray
from ray import tune
import numpy as np
import torch
from os import path
from data import get_data
from model import Net


def compute_val(a, b):
    return (a - 3) ** 2 + (b - 7) ** 3


data = [get_data() for _ in range(1)]


def trail(config, ckp_dir=None, data=None):
    tune.utils.wait_for_gpu(target_util=0.25)
    net = Net(256)
    assert data is not None
    a = torch.tensor([config["a"] + len(data)])
    b = torch.tensor([config["b"]])
    val = net.forward(a.unsqueeze(0), b.unsqueeze(0))
    tune.report(value=val.item())


search_space = {"a": tune.quniform(0, 15, 0.5), "b": tune.quniform(0, 15, 0.5)}


exp = tune.run(
    tune.with_parameters(trail, data=data),
    metric="value",
    mode="max",
    resources_per_trial={"cpu": 0.1, "gpu": 1},
    config=search_space,
    num_samples=100,
    verbose=1,
    local_dir=path.abspath("./ray"),
)

print(exp.best_config)
