import torch
from torch import nn

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
        ratio = 3 * 128 * 128 / latent
        print(f"base={base} latent={latent}: params {count(model) / 1e6:.1f}M, "
              f"latent {tuple(z.shape)}, output {tuple(out.shape)}, "
              f"range {out.min():.2f} to {out.max():.2f}, compression {ratio:.0f}x")
        assert out.shape == x.shape and z.shape == (4, latent)

    model = DenoisingAutoencoder(base=32, latent_dim=128, dropout=0.1).to(device)
    model.eval()
    with torch.no_grad():
        same = torch.equal(model(x), model(x))
    print("eval mode is deterministic:", same)

    model.train()
    batch = torch.rand(8, 3, 128, 128, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    first = None
    for step in range(60):
        optimizer.zero_grad()
        loss = loss_fn(model(batch), batch)
        loss.backward()
        optimizer.step()
        first = loss.item() if first is None else first
    print(f"overfit one batch: loss {first:.3f} -> {loss.item():.3f}")
    assert loss.item() < first


if __name__ == "__main__":
    main()
