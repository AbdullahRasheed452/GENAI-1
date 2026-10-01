import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.models.cgan import UNetGenerator
from src.training.train_cgan import FS2KDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    g = UNetGenerator(cfg["base"], cfg["embed_dim"], cfg["dropout"])
    g.load_state_dict(ckpt["generator"])
    g.eval()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        g,
        (torch.rand(1, 3, 128, 128), torch.zeros(1, dtype=torch.long)),
        args.out,
        input_names=["photo", "style"],
        output_names=["sketch"],
        dynamic_axes={"photo": {0: "batch"}, "style": {0: "batch"}, "sketch": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"exported {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB), checkpoint epoch {ckpt['epoch'] + 1}")

    test = FS2KDataset("test")
    idx = np.linspace(0, len(test) - 1, 32).astype(int)
    items = [test[int(i)] for i in idx]
    photo = torch.stack([it[0] for it in items])
    style = torch.tensor([it[2] for it in items], dtype=torch.long)
    with torch.no_grad():
        reference = g(photo, style).numpy()

    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    result = session.run(None, {"photo": photo.numpy(), "style": style.numpy()})[0]
    diff = np.abs(reference - result).max()
    print(f"batch of 32: output {result.shape}, max abs diff {diff:.2e}")
    assert diff < 1e-3, "ONNX output does not match PyTorch"

    one = {"photo": photo[:1].numpy(), "style": style[:1].numpy()}
    session.run(None, one)
    started = time.time()
    for _ in range(20):
        session.run(None, one)
    print(f"single image on CPU: {(time.time() - started) / 20 * 1000:.1f} ms")
    print("ONNX matches PyTorch")


if __name__ == "__main__":
    main()
