"""Seeded 100-epoch nnU-Net trainers for Protocol V3."""

import os
import random

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _seed_all(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _ProtocolV3SeededTrainer(nnUNetTrainer):
    protocol_seed = 123

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        _seed_all(self.protocol_seed)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100

    def on_train_start(self):
        _seed_all(self.protocol_seed)
        super().on_train_start()


class nnUNetTrainerProtocolV3Seed123(_ProtocolV3SeededTrainer):
    protocol_seed = 123


class nnUNetTrainerProtocolV3Seed456(_ProtocolV3SeededTrainer):
    protocol_seed = 456


class nnUNetTrainerProtocolV3Seed789(_ProtocolV3SeededTrainer):
    protocol_seed = 789
