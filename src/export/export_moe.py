import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.data.datasets import ManifestDataset
from src.models.moe import load_moe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = load_moe(args.checkpoint, "cpu").eval()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.rand(1, 3, 128, 128),
        args.out,
        input_names=["input"],
        output_names=["restored", "weights"],
        dynamic_axes={"input": {0: "batch"}, "restored": {0: "batch"}, "weights": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"exported {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")

    test = ManifestDataset("test")
    idx = np.linspace(0, len(test) - 1, 32).astype(int)
    x = torch.stack([test[int(i)][0] for i in idx])
    with torch.no_grad():
        restored, weights = model(x)
    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    out_restored, out_weights = session.run(None, {"input": x.numpy()})
    d_img = np.abs(restored.numpy() - out_restored).max()
    d_w = np.abs(weights.numpy() - out_weights).max()
    print(f"batch of 32: max abs diff image {d_img:.2e}, weights {d_w:.2e}, weight sums {out_weights.sum(1).min():.4f} to {out_weights.sum(1).max():.4f}")
    assert d_img < 1e-3 and d_w < 1e-3, "ONNX output does not match PyTorch"

    single = {"input": x[:1].numpy()}
    session.run(None, single)
    started = time.time()
    for _ in range(20):
        session.run(None, single)
    print(f"single image on CPU: {(time.time() - started) / 20 * 1000:.1f} ms")
    print("ONNX matches PyTorch")


if __name__ == "__main__":
    main()
