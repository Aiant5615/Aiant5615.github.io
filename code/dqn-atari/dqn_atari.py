"""Playing Atari with Deep Reinforcement Learning (DQN, Mnih et al., 2013) — a from-scratch PyTorch implementation on a
tiny in-file grid world.

What is implemented (section numbers follow the paper):
  * Q-network: a small CNN reads the "screen" (3 binary channels: agent, goal, pits) and outputs Q(s, a; theta)
    for every action in one forward pass                                                        (4.1)
  * experience replay: transitions e_t = (s_t, a_t, r_t, s_{t+1}) stored in D, minibatches sampled uniformly (4, Algorithm 1)
  * the loss L_i(theta_i) = E_{(s,a,r,s')~D}[ (r + gamma max_a' Q(s', a'; theta^-) - Q(s, a; theta_i))^2 ]
    with the target parameters theta^- held fixed while theta is updated (theta_{i-1} in the 2013 paper;
    a periodically copied target network as in the Nature 2015 version)                        (3, eq. 2)
  * epsilon-greedy behaviour policy, epsilon annealed linearly from 1.0 to 0.1, then fixed      (5)
  * reward clipping to [-1, 1]                                                                  (4.2)
  * Huber (error-clipped) loss and a greedy evaluation of the learned policy                    (Nature version)
Simplifications: a 6 x 6 grid world stands in for Atari (no frame skipping, no 4-frame stacking, no 84 x 84
preprocessing); 12k environment steps instead of 10M frames; Adam instead of RMSProp; the agent and goal positions
are random every episode so the network has to read them off the screen.

Run:  python dqn_atari.py        (CPU, about 30 s)
"""
import random, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); random.seed(0); np.random.seed(0)


# ───────────────────────── environment: a 6x6 grid rendered as a 3-channel screen ─────────────────────────
class GridWorld:
    """Reach the goal (+1, episode ends), avoid the pits (-1, episode ends), -0.05 per step, at most 30 steps.
    Actions: 0 up, 1 down, 2 left, 3 right.  The observation is the screen (3, N, N), not the coordinates."""
    N, PITS, MAX_STEPS = 6, [(1, 2), (2, 4), (3, 1), (4, 3)], 24
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def reset(self):
        free = [(r, c) for r in range(self.N) for c in range(self.N) if (r, c) not in self.PITS]
        self.agent, self.goal = random.sample(free, 2)
        self.t = 0
        return self.render()

    def render(self):
        screen = np.zeros((3, self.N, self.N), dtype=np.float32)
        screen[0][self.agent], screen[1][self.goal] = 1.0, 1.0
        for p in self.PITS:
            screen[2][p] = 1.0
        return screen

    def step(self, a):
        dr, dc = self.MOVES[a]
        r, c = self.agent
        self.agent = (min(max(r + dr, 0), self.N - 1), min(max(c + dc, 0), self.N - 1))   # walls: stay in place
        self.t += 1
        if self.agent == self.goal:
            return self.render(), 1.0, True
        if self.agent in self.PITS:
            return self.render(), -1.0, True
        return self.render(), -0.05, self.t >= self.MAX_STEPS


# ───────────────────────── Q-network (4.1) ─────────────────────────
class QNetwork(nn.Module):
    def __init__(self, n_actions=4, N=GridWorld.N):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 16, 3, padding=1)
        self.fc1, self.fc2 = nn.Linear(16 * N * N, 64), nn.Linear(64, n_actions)

    def forward(self, screen):                       # (B, 3, N, N) -> (B, n_actions): one Q-value per action
        x = F.relu(self.conv2(F.relu(self.conv1(screen))))
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


class ReplayMemory:
    """D = {e_1, ..., e_N}; minibatches of experience drawn uniformly at random (Algorithm 1)."""
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, B):
        s, a, r, s2, d = zip(*random.sample(self.buf, B))
        return (torch.from_numpy(np.stack(s)), torch.tensor(a), torch.tensor(r), torch.from_numpy(np.stack(s2)), torch.tensor(d, dtype=torch.float32))


def dqn_loss(q, q_target, batch, gamma):
    """eq. 2:  y = r + gamma max_a' Q(s', a'; theta^-)  (y = r at terminal s');  loss between y and Q(s, a; theta)."""
    s, a, r, s2, done = batch
    q_sa = q(s).gather(1, a[:, None]).squeeze(1)
    with torch.no_grad():
        y = r + gamma * (1 - done) * q_target(s2).max(1).values      # the target uses the old parameters theta^-
    return F.smooth_l1_loss(q_sa, y)                                  # Huber: squared for |error| < 1, linear beyond


def epsilon(step, start=1.0, end=0.1, anneal_steps=6000):
    return max(end, start + (end - start) * step / anneal_steps)      # linear anneal, then fixed at 0.1


@torch.no_grad()
def evaluate(q, episodes=200):
    env, returns, successes = GridWorld(), [], 0
    for _ in range(episodes):
        s, done, G = env.reset(), False, 0.0
        while not done:
            s, r, done = env.step(q(torch.from_numpy(s)[None]).argmax().item())     # greedy, epsilon = 0
            G += r
        returns.append(G); successes += r == 1.0
    return float(np.mean(returns)), successes / episodes


def main():
    t0 = time.time()
    env, memory = GridWorld(), ReplayMemory(10000)
    q, q_target = QNetwork(), QNetwork()
    q_target.load_state_dict(q.state_dict())                          # theta^- <- theta
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    gamma, B, total_steps, learn_start, target_every = 0.95, 32, 12000, 500, 250
    print(f"Q-network parameters: {sum(p.numel() for p in q.parameters()):,}")
    episode_returns, s, G = [], env.reset(), 0.0
    for step in range(1, total_steps + 1):
        if random.random() < epsilon(step):                           # epsilon-greedy behaviour policy
            a = random.randrange(4)
        else:
            with torch.no_grad():
                a = q(torch.from_numpy(s)[None]).argmax().item()
        s2, r, done = env.step(a)
        terminal = abs(r) == 1.0                                       # a time-out is not a terminal state
        memory.push(s, a, float(np.clip(r, -1, 1)), s2, terminal)    # reward clipping (4.2)
        G += r; s = s2
        if done:
            episode_returns.append(G); s, G = env.reset(), 0.0
        if step >= learn_start:
            loss = dqn_loss(q, q_target, memory.sample(B), gamma)
            opt.zero_grad(); loss.backward(); opt.step()
        if step % target_every == 0:
            q_target.load_state_dict(q.state_dict())                  # theta^- <- theta
        if step % 2000 == 0:
            recent = episode_returns[-50:]
            print(f"step {step:5d}  episodes {len(episode_returns):4d}  eps {epsilon(step):.2f}  loss {loss.item():.4f}"
                  f"  mean return (last 50 episodes) {np.mean(recent):6.3f}")
    first, last = np.mean(episode_returns[:50]), np.mean(episode_returns[-50:])
    mean_return, success = evaluate(q)
    print(f"training return: first 50 episodes {first:.3f} -> last 50 episodes {last:.3f}")
    print(f"greedy evaluation over 200 episodes: mean return {mean_return:.3f}, goal reached {success:.2f}   ({time.time() - t0:.1f} s)")
    assert last > first + 0.5 and success > 0.9, "DQN did not learn the grid world as expected"


if __name__ == "__main__":
    main()
