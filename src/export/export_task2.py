import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from src.data.datasets import ManifestDataset
from src.models.autoencoder import DenoisingAutoencoder
from src.models.classifier import CorruptionClassifier

KINDS = ("salt_pepper", "blur", "occlusion")


class WithSoftmax(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


def export_and_check(model, path, x, name):
    model.eval()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.rand(1, 3, 128, 128),
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    with torch.no_grad():
        reference = model(x).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    result = session.run(None, {"input": x.numpy()})[0]
    diff = np.abs(reference - result).max()
    single = x[:1].numpy()
    session.run(None, {"input": single})
    started = time.time()
    for _ in range(20):
        session.run(None, {"input": single})
    ms = (time.time() - started) / 20 * 1000
    print(f"{name}: {Path(path).stat().st_size / 1e6:.1f} MB, max abs diff vs PyTorch {diff:.2e}, {ms:.1f} ms per image")
    assert diff < 1e-3, f"{name}: ONNX output does not match PyTorch"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task2_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    test = ManifestDataset("test")
    idx = np.linspace(0, len(test) - 1, 64).astype(int)
    x = torch.stack([test[int(i)][0] for i in idx])
    src, out = Path(args.task2_dir), Path(args.out_dir)

    ckpt = torch.load(src / "classifier_final.pt", map_location="cpu")
    cfg = ckpt["config"]
    classifier = CorruptionClassifier(cfg["channels"], cfg["dropout"])
    classifier.load_state_dict(ckpt["model"])
    export_and_check(WithSoftmax(classifier), out / "task2_classifier.onnx", x, "classifier")

    for kind in KINDS:
        ckpt = torch.load(src / f"specialist_{kind}.pt", map_location="cpu")
        cfg = ckpt["config"]
        model = DenoisingAutoencoder(cfg["base"], cfg["latent_dim"], cfg["dropout"])
        model.load_state_dict(ckpt["model"])
        export_and_check(model, out / f"task2_specialist_{kind}.onnx", x, f"specialist {kind} (epoch {ckpt['epoch'] + 1})")
    print("all four Task 2 models exported and verified")


if __name__ == "__main__":
    main()
