"""RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (Brohan et al., 2023) — toy implementation.

What is implemented (section numbers follow the paper):
  * actions as text: the 8-integer action string "terminate dx dy dz droll dpitch dyaw gripper",
    each continuous dimension discretised into 256 bins and emitted as integer tokens of the VLM      (Sec. 3.1)
  * a toy VLM: a decoder-only Transformer over [image tokens, question tokens, answer tokens]
    whose vocabulary already contains one token per integer 0..255 (the PaLI-X case)                  (Sec. 3.1)
  * co-fine-tuning: the pretrained VLM is fine-tuned on a mixture of its web (VQA) data and robot data,
    with the robot fraction increased over training, versus fine-tuning on robot data only            (Sec. 3.1, 5.4)
  * constrained decoding: at action positions the sampler may only emit valid action tokens          (Sec. 3.1)
  * chain-of-thought actions: the policy first emits the object's cell (a short "plan" in the tokens the web
    "where" question uses) and then the 8 action integers                                             (Sec. 4.4)
  * the transfer effect: robot data names only half of the objects; web VQA data grounds all of them,
    so the model acts correctly on objects it never saw in robot data ("unseen objects", Sec. 5.1); co-fine-tuning
    keeps the web skill (VQA accuracy) that robot-only fine-tuning forgets                              (Sec. 5.4)
Simplifications: a 3x3 symbolic image (one token per cell) instead of camera pixels; a 2-layer Transformer instead
of PaLI-X 55B; "web data" is a toy VQA task ("where is X?" / "what is at P?") on the same scenes; the robot task is
one step of a scripted pick policy (dx, dy toward the object, close the gripper on arrival).

Run:  python rt2.py        (CPU, about 40 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ───────────────────────── vocabulary ─────────────────────────
PAD, BOS, EOS, Q_WHERE, Q_WHAT, Q_PICK, EMPTY = range(7)
N_OBJ, N_CELL, BINS = 12, 9, 256
OBJ0, POS0, INT0 = 7, 7 + N_OBJ, 7 + N_OBJ + N_CELL                 # object tokens, cell tokens, integer tokens 0..255
VOCAB = INT0 + BINS
SEEN, UNSEEN = list(range(6)), list(range(6, 12))                   # objects that appear in robot data / only in web data
ACTION_LEN = 8
PROMPT_LEN = 1 + N_CELL + 2                                         # BOS img(9) Q obj
SEQ_LEN = PROMPT_LEN + 1 + ACTION_LEN + 1                           # ... plan(1) action(8) EOS


def discretise(x):
    """One continuous dimension in [-1, 1] -> a bin in 0..255 (Sec. 3.1)."""
    return int(round((x + 1) / 2 * (BINS - 1)))


def make_scene():
    """3 distinct objects on a 3x3 grid; the gripper hovers over the centre cell (index 4)."""
    objs = torch.randperm(N_OBJ)[:3].tolist()
    cells = torch.randperm(N_CELL)[:3].tolist()
    img = [EMPTY] * N_CELL
    for o, c in zip(objs, cells):
        img[c] = OBJ0 + o
    return img, dict(zip(objs, cells))


def expert_action(cell):
    """Scripted pick policy as an RT-2 action string: terminate, dx, dy, dz, droll, dpitch, dyaw, gripper."""
    dx, dy = (cell % 3) - 1, (cell // 3) - 1                          # cell offset from the gripper (centre), in {-1, 0, 1}
    at_object = dx == 0 and dy == 0
    a = [1 if at_object else 0, discretise(dx), discretise(dy), discretise(-0.5 if at_object else 0.0),
         discretise(0.0), discretise(0.0), discretise(0.0), discretise(1.0 if at_object else -1.0)]
    return a


def make_batch(B, robot, objects=None):
    """Web example: [BOS, img, Q_WHERE, obj, pos, EOS] or [BOS, img, Q_WHAT, pos, obj, EOS].
    Robot example: [BOS, img, Q_PICK, obj, pos, a_1..a_8, EOS] — the policy first states the object's cell in the same
    tokens the web "where" question uses (the paper's chain-of-thought variant, Sec. 4.4: a short plan before the
    action), then the 8 action integers. The loss is only on the answer tokens after the prompt."""
    seqs = torch.full((B, SEQ_LEN), PAD)
    for i in range(B):
        img, where = make_scene()
        pool = [o for o in where if objects is None or o in objects] or list(where)
        o = pool[torch.randint(len(pool), (1,)).item()]
        if robot:
            ans = [POS0 + where[o]] + [INT0 + t for t in expert_action(where[o])]
            q = [Q_PICK, OBJ0 + o]
        elif torch.rand(()) < 0.5:
            ans, q = [POS0 + where[o]], [Q_WHERE, OBJ0 + o]
        else:
            ans, q = [OBJ0 + o], [Q_WHAT, POS0 + where[o]]
        s = [BOS] + img + q + ans + [EOS]
        seqs[i, : len(s)] = torch.tensor(s)
    return seqs


# ───────────────────────── toy VLM ─────────────────────────
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.ln1, self.ln2 = h, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2) / math.sqrt(d // self.h)).masked_fill(~mask, float("-inf")).softmax(-1)
        x = x + self.out((att @ v).transpose(1, 2).reshape(B, L, d))
        return x + self.mlp(self.ln2(x))


class ToyVLM(nn.Module):
    """Decoder-only Transformer: image tokens, question tokens and answer tokens all live in one vocabulary."""
    def __init__(self, d=64, h=4, n_layers=2):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB, d), nn.Parameter(torch.randn(SEQ_LEN, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, VOCAB)

    def forward(self, seq):
        L = seq.shape[1]
        x = self.tok(seq) + self.pos[:L]
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln(x))


def loss_fn(model, seq):
    """Next-token cross-entropy on the answer tokens only (everything after the prompt)."""
    logits = model(seq[:, :-1])
    tgt = seq[:, 1:].clone()
    tgt[:, : PROMPT_LEN - 1] = PAD
    return F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=PAD)


@torch.no_grad()
def decode_action(model, prompt):
    """Constrained greedy decoding (Sec. 3.1): a cell token for the plan step, then only the 256 integer tokens at the
    8 action positions."""
    seq = prompt
    plan = torch.full((VOCAB,), float("-inf")); plan[POS0:INT0] = 0.0
    allowed = torch.full((VOCAB,), float("-inf")); allowed[INT0:] = 0.0
    for i in range(1 + ACTION_LEN):
        nxt = (model(seq)[:, -1] + (plan if i == 0 else allowed)).argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], 1)
    return seq[:, -ACTION_LEN:] - INT0                                 # (B, 8) integer bins


@torch.no_grad()
def action_accuracy(model, objects, n=200):
    seq = make_batch(n, robot=True, objects=objects)
    pred = decode_action(model, seq[:, :PROMPT_LEN])
    gold = seq[:, PROMPT_LEN + 1 : PROMPT_LEN + 1 + ACTION_LEN] - INT0
    return (pred == gold).all(1).float().mean().item()


@torch.no_grad()
def vqa_accuracy(model, n=200):
    seq = make_batch(n, robot=False)
    pred = model(seq[:, :-1])[:, PROMPT_LEN - 1].argmax(-1)           # the first token after the prompt is the answer
    return (pred == seq[:, PROMPT_LEN]).float().mean().item()


def train(model, steps, robot_frac, log_every, tag):
    """robot_frac: callable step -> fraction of robot examples in the batch (0 = web only, 1 = robot only)."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100))
    for step in range(1, steps + 1):
        f = robot_frac(step / steps)
        n_robot = int(round(64 * f))
        batch = torch.cat([make_batch(n_robot, robot=True, objects=SEEN), make_batch(64 - n_robot, robot=False)])
        loss = loss_fn(model, batch)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % log_every == 0:
            print(f"  {tag} step {step:4d}  robot fraction {f:.2f}  loss {loss.item():.3f}")
    return loss.item()


def main():
    t0 = time.time()
    print(f"vocabulary: {VOCAB} tokens, of which 256 are the integer tokens 0..255 used for action bins")
    vlm = ToyVLM()
    print(f"parameters: {sum(p.numel() for p in vlm.parameters()):,}")
    print("1) pretrain the VLM on web (VQA) data only")
    train(vlm, 1200, lambda t: 0.0, 600, "web")
    print(f"   VQA accuracy {vqa_accuracy(vlm):.2f}")
    ft, coft = ToyVLM(), ToyVLM()
    ft.load_state_dict(vlm.state_dict()); coft.load_state_dict(vlm.state_dict())
    print("2a) fine-tune on robot data only (seen objects)")
    train(ft, 800, lambda t: 1.0, 400, "ft")
    print("2b) co-fine-tune on web + robot data, robot fraction ramped 0.25 -> 0.75 (Sec. 3.1)")
    train(coft, 800, lambda t: 0.25 + 0.5 * t, 400, "co-ft")
    ft.eval(); coft.eval()
    res = {name: (action_accuracy(m, SEEN), action_accuracy(m, UNSEEN), vqa_accuracy(m)) for name, m in [("fine-tune", ft), ("co-fine-tune", coft)]}
    print(f"{'model':14s} {'seen-object acc':>16s} {'unseen-object acc':>18s} {'web VQA acc':>12s}")
    for name, (s, u, v) in res.items():
        print(f"{name:14s} {s:16.2f} {u:18.2f} {v:12.2f}")
    seq = make_batch(1, robot=True, objects=UNSEEN)
    bins = decode_action(coft, seq[:, :PROMPT_LEN])[0].tolist()
    cont = [b / (BINS - 1) * 2 - 1 for b in bins[1:]]
    print(f"example (unseen object {seq[0, PROMPT_LEN - 1].item() - OBJ0}): action string {' '.join(map(str, bins))}")
    print(f"  -> terminate={bins[0]}, (dx, dy, dz, droll, dpitch, dyaw, gripper) = {[round(c, 2) for c in cont]}   ({time.time() - t0:.1f} s)")
    s_ft, u_ft, v_ft = res["fine-tune"]; s_co, u_co, v_co = res["co-fine-tune"]
    assert s_co > 0.9 and u_co > 0.9, "web-grounded objects did not transfer to robot actions"
    assert v_co > v_ft + 0.2, "co-fine-tuning did not preserve the web skill better than robot-only fine-tuning"


if __name__ == "__main__":
    main()
