"""BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2018) — a from-scratch PyTorch implementation.

What is implemented (section numbers follow the paper):
  * Transformer encoder with bidirectional (fully-visible) self-attention, post-LN, GELU        (3, Figure 3)
  * input representation E_i = E^token_{w_i} + E^segment_{s_i} + E^position_i, sequences packed as
    [CLS] A [SEP] B [SEP] with two learned segment embeddings E_A, E_B                           (3, Figure 2)
  * Task #1, masked LM: 15% of positions chosen; of those 80% -> [MASK], 10% -> random token,
    10% unchanged; cross-entropy on the chosen positions only                                     (3.1, Appendix C.2)
  * Task #2, next-sentence prediction: 50% IsNext / 50% NotNext, linear layer + softmax on C,
    the final hidden vector of [CLS]                                                                (3.1)
  * fine-tuning: a new W in R^{K x H}, log softmax(C W^T), all parameters updated                (3.2, 4.1, Figure 4)
  * control: the same fine-tuning from random initialisation
Simplifications: L = 2, H = 64, A = 4 instead of 12 / 768 / 12; a toy corpus of integer tokens (each document is a
Markov chain over the words of one of 8 hidden topics, a "sentence" is an 8-token chunk) instead of Wikipedia +
BooksCorpus, no WordPiece; the downstream task is topic classification of a single sentence from 64 labeled
examples; no dropout, no warm-up, Adam without weight decay.

Run:  python bert.py        (CPU, about 15 s)
"""
import math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); random.seed(0)

PAD, CLS, SEP, MASK = 0, 1, 2, 3                # special tokens; words are 4 .. 4 + V_TEXT - 1
V_TEXT, N_TOPICS, SENT_LEN = 40, 8, 8
VOCAB = 4 + V_TEXT


# ───────────────────────── encoder ─────────────────────────
class SelfAttention(nn.Module):
    """Bidirectional multi-head attention: every position attends to every non-pad position (Figure 3, "BERT")."""
    def __init__(self, H, A):
        super().__init__()
        self.A, self.d_k = A, H // A
        self.qkv, self.proj = nn.Linear(H, 3 * H), nn.Linear(H, H)

    def forward(self, x, pad_mask):                        # pad_mask: (B, L) True = real token
        B, L, _ = x.shape
        q, k, v = (t.transpose(1, 2) for t in self.qkv(x).view(B, L, 3, self.A, self.d_k).unbind(2))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = scores.masked_fill(~pad_mask[:, None, None, :], float("-inf")).softmax(-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, L, -1))


class EncoderLayer(nn.Module):
    def __init__(self, H, A, d_ff):
        super().__init__()
        self.attn, self.ln1 = SelfAttention(H, A), nn.LayerNorm(H)
        self.ffn, self.ln2 = nn.Sequential(nn.Linear(H, d_ff), nn.GELU(), nn.Linear(d_ff, H)), nn.LayerNorm(H)

    def forward(self, x, pad_mask):
        x = self.ln1(x + self.attn(x, pad_mask))
        return self.ln2(x + self.ffn(x))


class BERT(nn.Module):
    def __init__(self, vocab, max_len=64, H=64, A=4, L=2, n_classes=N_TOPICS):
        super().__init__()
        self.tok, self.seg, self.pos = nn.Embedding(vocab, H), nn.Embedding(2, H), nn.Embedding(max_len, H)
        self.emb_ln = nn.LayerNorm(H)
        self.layers = nn.ModuleList([EncoderLayer(H, A, 4 * H) for _ in range(L)])
        self.mlm_head = nn.Linear(H, vocab)                 # Task #1 output softmax over the vocabulary
        self.nsp_head = nn.Linear(H, 2)                     # Task #2: IsNext / NotNext from C
        self.cls_head = nn.Linear(H, n_classes)             # fine-tuning: W in R^{K x H} on C (4.1)

    def forward(self, tokens, segments):
        """E_i = E^token + E^segment + E^position, then L encoder layers. Returns all final vectors T_i; T_0 = C."""
        pad_mask = tokens != PAD
        x = self.emb_ln(self.tok(tokens) + self.seg(segments) + self.pos(torch.arange(tokens.size(1))))
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x


# ───────────────────────── toy corpus ─────────────────────────
# Topic t owns 5 words arranged in a ring; the next word is the ring successor (p = 0.5), another topic word (0.3)
# or any word (0.2).  A masked word is therefore predictable from both its left and its right neighbour.
TOPIC_WORDS = torch.stack([torch.randperm(V_TEXT)[:5] for _ in range(N_TOPICS)]) + 4


def sample_document(topic, n_tokens):
    words = TOPIC_WORDS[topic].tolist()
    doc, w = [], random.choice(words)
    for _ in range(n_tokens):
        doc.append(w)
        r = random.random()
        if r < 0.5 and w in words: w = words[(words.index(w) + 1) % len(words)]     # ring successor
        elif r < 0.8:              w = random.choice(words)                          # another word of the topic
        else:                      w = random.randrange(4, VOCAB)                    # any word
    return doc


def make_pretrain_batch(B):
    """Sentence pairs: [CLS] A [SEP] B [SEP].  50% B follows A in the same document (IsNext), 50% B is a random
    sentence from another document (NotNext) (3.1, Task #2).  Segment ids 0 for [CLS] A [SEP], 1 for B [SEP]."""
    tokens, segments, is_next = [], [], []
    for _ in range(B):
        topic = torch.randint(0, N_TOPICS, (1,)).item()
        doc = sample_document(topic, 2 * SENT_LEN)
        A, Bs = doc[:SENT_LEN], doc[SENT_LEN:]
        label = torch.randint(0, 2, (1,)).item()
        if label == 0:                                              # NotNext: a sentence from some other document
            Bs = sample_document(torch.randint(0, N_TOPICS, (1,)).item(), SENT_LEN)
        tokens.append([CLS] + A + [SEP] + Bs + [SEP])
        segments.append([0] * (SENT_LEN + 2) + [1] * (SENT_LEN + 1))
        is_next.append(label)
    return torch.tensor(tokens), torch.tensor(segments), torch.tensor(is_next)


def mask_tokens(tokens):
    """Task #1 masking (3.1).  15% of the (non-special) positions are chosen; labels are -100 elsewhere so that the
    loss only covers the chosen positions.  Of the chosen positions: 80% [MASK], 10% random word, 10% unchanged."""
    labels = tokens.clone()
    chosen = (torch.rand(tokens.shape) < 0.15) & (tokens >= 4)
    labels[~chosen] = -100
    r = torch.rand(tokens.shape)
    inp = tokens.clone()
    inp[chosen & (r < 0.8)] = MASK
    random_words = torch.randint(4, VOCAB, tokens.shape)
    swap = chosen & (r >= 0.8) & (r < 0.9)
    inp[swap] = random_words[swap]                                    # the remaining 10% keep the original token
    return inp, labels


def make_cls_batch(B):
    """Single-sentence classification input: [CLS] sentence [SEP], all segment 0; the label is the hidden topic."""
    topics = torch.randint(0, N_TOPICS, (B,))
    tokens = torch.tensor([[CLS] + sample_document(t.item(), SENT_LEN) + [SEP] for t in topics])
    return tokens, torch.zeros_like(tokens), topics


def finetune(model, train, steps=100, lr=1e-3, tag=""):
    """Fine-tune all parameters with the classification loss on C (4.1); the paper uses 2-4 epochs at lr 2e-5 to 5e-5."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tokens, segments, y = train
    for step in range(1, steps + 1):
        loss = F.cross_entropy(model.cls_head(model(tokens, segments)[:, 0]), y)
        opt.zero_grad(); loss.backward(); opt.step()
    tokens, segments, y = make_cls_batch(500)
    with torch.no_grad():
        acc = (model.cls_head(model(tokens, segments)[:, 0]).argmax(-1) == y).float().mean().item()
    print(f"  [{tag}] fine-tune loss {loss.item():.3f}  held-out topic accuracy {acc:.2f}")
    return acc


def main():
    t0 = time.time()
    model = BERT(VOCAB)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("pre-training: masked LM + next-sentence prediction")
    for step in range(1, 601):
        tokens, segments, is_next = make_pretrain_batch(32)
        inp, labels = mask_tokens(tokens)
        T = model(inp, segments)
        mlm = F.cross_entropy(model.mlm_head(T).view(-1, VOCAB), labels.view(-1), ignore_index=-100)
        nsp = F.cross_entropy(model.nsp_head(T[:, 0]), is_next)         # C = T_0, the [CLS] vector
        loss = mlm + nsp
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1:
            first_mlm = mlm.item()
        if step % 150 == 0:
            nsp_acc = (model.nsp_head(T[:, 0]).argmax(-1) == is_next).float().mean().item()
            print(f"  step {step:3d}  MLM loss {mlm.item():.3f}  NSP loss {nsp.item():.3f}  NSP acc {nsp_acc:.2f}")
    with torch.no_grad():
        tokens, segments, is_next = make_pretrain_batch(500)
        nsp_acc = (model.nsp_head(model(tokens, segments)[:, 0]).argmax(-1) == is_next).float().mean().item()
    print(f"held-out NSP accuracy {nsp_acc:.2f}  (chance 0.50; NotNext from the same topic is undetectable, so the ceiling is ~0.94)")
    print("fine-tuning [CLS] for topic classification from 64 labeled sentences")
    train = make_cls_batch(64)
    acc_pre = finetune(model, train, tag="pre-trained")
    acc_scratch = finetune(BERT(VOCAB), train, tag="no pre-training")
    print(f"done in {time.time() - t0:.1f} s")
    assert mlm.item() < 0.7 * first_mlm and nsp_acc > 0.75 and acc_pre > 0.85 and acc_pre >= acc_scratch, \
        "pre-training or fine-tuning did not work as expected"


if __name__ == "__main__":
    main()
