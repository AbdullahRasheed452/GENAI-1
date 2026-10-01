import numpy as np
import torch
from torch import nn

from src.data.datasets import CACHE, to_tensor
from src.models.autoencoder import DenoisingAutoencoder


def count(model):
    return sum(p.numel() for p in model.parameters())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(4, 3, 128, 128, device=device)

    for base, latent in [(32, 128), (64, 256), (64, 512)]:
        model = DenoisingAutoencoder(base=base, latent_dim=latent, dropout=0.1).to(device)
        z = model.encode(x)
        out = model(x)
        print(f"base={base} latent={latent}: params {count(model) / 1e6:.1f}M, "
              f"latent {tuple(z.shape)}, output {tuple(out.shape)}, "
              f"compression {3 * 128 * 128 / latent:.0f}x")
        assert out.shape == x.shape and z.shape == (4, latent)

    model = DenoisingAutoencoder(base=32, latent_dim=128, dropout=0.0).to(device)
    model.eval()
    with torch.no_grad():
        print("eval mode is deterministic:", torch.equal(model(x), model(x)))

    images = np.load(CACHE / "train.npy", mmap_mode="r")[:8]
    batch = torch.stack([to_tensor(np.array(i)) for i in images]).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    for step in range(201):
        optimizer.zero_grad()
        loss = loss_fn(model(batch), batch)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"overfit 8 real images, step {step}: L1 loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
