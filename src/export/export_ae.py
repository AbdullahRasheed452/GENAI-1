import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.data.datasets import ManifestDataset
from src.models.autoencoder import DenoisingAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = DenoisingAutoencoder(cfg["base"], cfg["latent_dim"], cfg["dropout"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.rand(1, 3, 128, 128),
        args.out,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"exported {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")

    test = ManifestDataset("test")
    idx = np.linspace(0, len(test) - 1, 64).astype(int)
    x = torch.stack([test[int(i)][0] for i in idx])
    with torch.no_grad():
        reference = model(x).numpy()

    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    result = session.run(None, {"input": x.numpy()})[0]
    diff = np.abs(reference - result)
    print(f"batch of 64: output {result.shape}, max abs diff {diff.max():.2e}, mean abs diff {diff.mean():.2e}")
    assert diff.max() < 1e-3, "ONNX output does not match PyTorch"

    single = x[:1].numpy()
    session.run(None, {"input": single})
    started = time.time()
    for _ in range(20):
        session.run(None, {"input": single})
    print(f"single image on CPU: {(time.time() - started) / 20 * 1000:.1f} ms")
    print("ONNX matches PyTorch")


if __name__ == "__main__":
    main()
