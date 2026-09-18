"""Deep Residual Learning for Image Recognition (He, Zhang, Ren & Sun, 2016) — from-scratch PyTorch.

What is implemented (section numbers follow the paper):
  * residual building block  y = F(x, {W_i}) + x  with F = two 3x3 conv-BN-ReLU layers and a ReLU after the add  (3.2, eq. 1; Fig. 2)
  * shortcut options when dimensions change (3.3, 4.1): (A) identity with zero-padded channels and stride-2
    subsampling, (B) 1x1 projection  y = F(x) + W_s x   (eq. 2)
  * bottleneck block 1x1 -> 3x3 -> 1x1 for deeper nets (4.1, Fig. 5)
  * the CIFAR-style network of Section 4.2: 3x3 conv, three stages of 2n blocks with {16, 32, 64} filters, feature
    map halved and filters doubled at each stage, global average pooling, linear classifier -> 6n + 2 layers
  * the degradation experiment (Section 3.1, Fig. 1 / Fig. 6): the same 6n+2 architecture with and without shortcuts,
    trained identically; He initialisation, BN, SGD with momentum 0.9 and weight decay 1e-4 (4.2)
  * Figure 7: standard deviation of the residual branch outputs F(x) vs. the block inputs after training
Simplifications: 12x12 synthetic "shape" images with 6 classes instead of CIFAR-10, n = 5 (32 layers) and a few
hundred SGD steps instead of 64k iterations, no data augmentation, no learning-rate schedule.

Run:  python resnet.py        (CPU, about 35 s)
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def conv3x3(c_in, c_out, stride=1):
    return nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)


class Shortcut(nn.Module):
    """Identity when shapes match; otherwise option A (subsample + zero-pad channels, no parameters) or B (1x1 projection W_s)."""
    def __init__(self, c_in, c_out, stride, option="A"):
        super().__init__()
        self.needs, self.option, self.pad = (c_in != c_out or stride != 1), option, c_out - c_in
        self.stride = stride
        if self.needs and option == "B":
            self.proj = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        if not self.needs:
            return x                                                      # y = F(x) + x           eq. 1
        if self.option == "B":
            return self.proj(x)                                           # y = F(x) + W_s x       eq. 2
        return F.pad(x[:, :, :: self.stride, :: self.stride], (0, 0, 0, 0, 0, self.pad))   # option A: zero channels


class BasicBlock(nn.Module):
    """F(x) = BN(W_2 relu(BN(W_1 x))), two 3x3 convs;  y = relu(F(x) + shortcut(x))   (Fig. 2)."""
    def __init__(self, c_in, c_out, stride=1, residual=True, option="A"):
        super().__init__()
        self.residual = residual
        self.f = nn.Sequential(conv3x3(c_in, c_out, stride), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                               conv3x3(c_out, c_out), nn.BatchNorm2d(c_out))
        self.shortcut = Shortcut(c_in, c_out, stride, option)
        self.last_F, self.last_x = None, None                             # for the Figure 7 response statistics

    def forward(self, x):
        Fx = self.f(x)
        if not self.residual:
            return F.relu(Fx)                                              # plain network: y = relu(F(x)), no shortcut
        self.last_F, self.last_x = Fx.detach(), x.detach()
        return F.relu(Fx + self.shortcut(x))


class Bottleneck(nn.Module):
    """1x1 (reduce to c_mid) -> 3x3 -> 1x1 (restore 4 c_mid); the 3x3 conv runs at reduced width (Fig. 5, right)."""
    def __init__(self, c_in, c_mid, stride=1, option="B"):
        super().__init__()
        c_out = 4 * c_mid
        self.f = nn.Sequential(nn.Conv2d(c_in, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
                               conv3x3(c_mid, c_mid, stride), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
                               nn.Conv2d(c_mid, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
        self.shortcut = Shortcut(c_in, c_out, stride, option)

    def forward(self, x):
        return F.relu(self.f(x) + self.shortcut(x))


class CifarResNet(nn.Module):
    """6n + 2 layers: conv3x3(16) -> n blocks x {16, 32, 64} filters (stride 2 at stage entry) -> global avg pool -> fc."""
    def __init__(self, n, n_classes, residual=True, option="A", in_ch=1):
        super().__init__()
        self.stem = nn.Sequential(conv3x3(in_ch, 16), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        blocks, c = [], 16
        for stage, width in enumerate((16, 32, 64)):
            for b in range(n):
                blocks.append(BasicBlock(c, width, stride=2 if (stage > 0 and b == 0) else 1, residual=residual, option=option))
                c = width
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(64, n_classes)
        for m in self.modules():                                           # He initialisation (Section 3.4 / 4.2)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.fc(self.blocks(self.stem(x)).mean((2, 3)))


# ───────────────────────── synthetic images ─────────────────────────
def make_images(B, size=12, n_classes=6):
    """Class k = one of 6 shapes (bar orientations, a square, a cross, a diagonal, a ring) drawn at a random position with
    random contrast on a noisy background. Needs several layers of translation-invariant features to classify."""
    y = torch.randint(n_classes, (B,))
    img = torch.randn(B, 1, size, size) * 0.5
    for i in range(B):
        r, c = torch.randint(1, size - 5, (2,)).tolist()
        s = torch.zeros(5, 5)
        k = y[i].item()
        if k == 0: s[2, :] = 1                                             # horizontal bar
        elif k == 1: s[:, 2] = 1                                           # vertical bar
        elif k == 2: s[1:4, 1:4] = 1                                       # square
        elif k == 3: s[2, :] = 1; s[:, 2] = 1                              # cross
        elif k == 4: s[torch.arange(5), torch.arange(5)] = 1               # diagonal
        else: s[[0, 0, 4, 4, 0, 4, 1, 2, 3, 1, 2, 3], [0, 4, 0, 4, 1, 1, 0, 0, 0, 4, 4, 4]] = 1; s[0, 2] = s[4, 2] = s[0, 3] = s[4, 3] = 1  # ring
        img[i, 0, r : r + 5, c : c + 5] += s * (1.5 + torch.rand(1).item())
    return img, y


def train(model, steps, tag):
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)   # Section 4.2 (lr scaled down)
    losses = []
    for step in range(1, steps + 1):
        x, y = make_images(64)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % (steps // 4) == 0:
            print(f"  {tag:10s} step {step:4d}  train loss {sum(losses[-25:]) / 25:.3f}")
    return sum(losses[-25:]) / 25


@torch.no_grad()
def accuracy(model, n=512):
    model.eval()
    x, y = make_images(n)
    acc = (model(x).argmax(1) == y).float().mean().item()
    model.train()
    return acc


def main():
    t0 = time.time()
    n, steps = 5, 300                                                     # 6n + 2 = 32 layers
    res = {}
    for residual in (False, True):
        torch.manual_seed(1)
        model = CifarResNet(n, n_classes=6, residual=residual)
        tag = "ResNet-32" if residual else "plain-32"
        print(f"{tag}: {sum(p.numel() for p in model.parameters()):,} parameters, {6 * n + 2} weighted layers")
        final = train(model, steps, tag)
        acc = accuracy(model)
        res[residual] = (final, acc, model)
        print(f"{tag}: final train loss {final:.3f}, held-out accuracy {acc:.2f}   ({time.time() - t0:.1f} s)")

    # Figure 7: residual functions stay small relative to their inputs
    model = res[True][2]
    with torch.no_grad():
        model.eval(); model(make_images(128)[0]); model.train()
    ratios = [b.last_F.std().item() / b.last_x.std().item() for b in model.blocks]
    print("std(F(x)) / std(x) per residual block:", " ".join(f"{r:.2f}" for r in ratios))

    # bottleneck block and projection shortcut: shape check (Section 4.1)
    x = torch.randn(2, 16, 12, 12)
    print("bottleneck block 16 -> (1x1 8, 3x3 8, 1x1 32), stride 2, option B:", tuple(Bottleneck(16, 8, stride=2)(x).shape))
    print("basic block 16 -> 32, stride 2, option A (zero-pad) / B (projection):",
          tuple(BasicBlock(16, 32, 2, option="A")(x).shape), tuple(BasicBlock(16, 32, 2, option="B")(x).shape))
    assert res[True][0] < res[False][0] and res[True][1] > res[False][1], "the residual net did not train better than the plain net"
    assert res[True][1] > 0.8, "ResNet did not learn the toy classification"


if __name__ == "__main__":
    main()
