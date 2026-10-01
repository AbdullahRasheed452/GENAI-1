from pathlib import Path

import torch
from torch import nn

from src.models.autoencoder import DenoisingAutoencoder
from src.models.classifier import CorruptionClassifier

KINDS = ("salt_pepper", "blur", "occlusion")


class SoftMoE(nn.Module):
    """Gate over [identity, salt_pepper, blur, occlusion]; output is the weighted sum of the branches."""

    def __init__(self, gate, experts, temperature=1.0):
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.temperature = temperature

    def forward(self, x, return_all=False):
        logits = self.gate(x)
        weights = torch.softmax(logits / self.temperature, dim=1)
        outputs = [x] + [expert(x) for expert in self.experts]
        y = sum(weights[:, i, None, None, None] * out for i, out in enumerate(outputs))
        if return_all:
            return y, weights, logits
        return y, weights


def build_architecture(gate_cfg, expert_cfgs, temperature):
    gate = CorruptionClassifier(gate_cfg["channels"], gate_cfg["dropout"])
    experts = [DenoisingAutoencoder(c["base"], c["latent_dim"], c["dropout"]) for c in expert_cfgs]
    return SoftMoE(gate, experts, temperature)


def build_moe(task2_dir, temperature, device):
    """Gate from the trained classifier, experts from the three trained specialists."""
    root = Path(task2_dir)
    gate_ckpt = torch.load(root / "classifier_final.pt", map_location="cpu")
    expert_cfgs, expert_states = [], []
    for kind in KINDS:
        ckpt = torch.load(root / f"specialist_{kind}.pt", map_location="cpu")
        expert_cfgs.append(ckpt["config"])
        expert_states.append(ckpt["model"])
    model = build_architecture(gate_ckpt["config"], expert_cfgs, temperature)
    model.gate.load_state_dict(gate_ckpt["model"])
    for expert, state in zip(model.experts, expert_states):
        expert.load_state_dict(state)
    model.gate_cfg, model.expert_cfgs = gate_ckpt["config"], expert_cfgs
    return model.to(device)


def load_moe(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = build_architecture(ckpt["gate_cfg"], ckpt["expert_cfgs"], ckpt["config"]["temperature"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()
