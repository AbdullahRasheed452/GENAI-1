import torch
from torch import nn

from src.models.cgan import NUM_STYLES, PatchDiscriminator, UNetGenerator


def count(model):
    return sum(p.numel() for p in model.parameters())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    photo = torch.rand(8, 3, 128, 128, device=device) * 2 - 1
    sketch = torch.rand(8, 3, 128, 128, device=device) * 2 - 1
    style = torch.arange(8, device=device) % NUM_STYLES

    for base, embed in [(32, 8), (64, 16)]:
        g = UNetGenerator(base, embed, 0.3).to(device)
        d = PatchDiscriminator(base, embed).to(device)
        fake = g(photo, style)
        patch = d(photo, style, fake)
        print(f"base={base} embed={embed}: generator {count(g) / 1e6:.1f}M params, discriminator {count(d) / 1e6:.1f}M params, "
              f"output {tuple(fake.shape)} range {fake.min():.2f} to {fake.max():.2f}, patch map {tuple(patch.shape)}")
        assert fake.shape == photo.shape

    g = UNetGenerator(32, 8, 0.3).to(device)
    d = PatchDiscriminator(32, 8).to(device)
    g.eval()
    with torch.no_grad():
        a = g(photo[:1], torch.tensor([0], device=device))
        b = g(photo[:1], torch.tensor([1], device=device))
        print("different styles give different outputs:", not torch.equal(a, b))
        print("eval mode is deterministic:", torch.equal(g(photo[:1], style[:1]), g(photo[:1], style[:1])))

    g.train()
    d.train()
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
    fake = g(photo, style)
    real_logits = d(photo, style, sketch)
    fake_logits = d(photo, style, fake.detach())
    d_loss = bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits))
    d_loss.backward()
    g_logits = d(photo, style, fake)
    g_loss = bce(g_logits, torch.ones_like(g_logits)) + 100 * l1(fake, sketch)
    g_loss.backward()
    print(f"one training step: D loss {d_loss.item():.3f}, G loss {g_loss.item():.3f}")
    print("gradient reaches generator style embedding:", bool(g.embed.weight.grad.abs().sum() > 0))
    print("gradient reaches discriminator style embedding:", bool(d.embed.weight.grad.abs().sum() > 0))


if __name__ == "__main__":
    main()
