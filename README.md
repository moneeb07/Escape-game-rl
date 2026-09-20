# RL Escape Room with Adversarial AI

A three-level escape room in which every level is played by an **autonomous agent**, without human input. Each level demonstrates a different branch of AI: deep reinforcement learning, adversarial multi-agent planning, and classical graph search under hazard constraints.

The agent must survive hazards, solve a randomized puzzle against a saboteur, and navigate a trap-filled maze. Levels are gated — failing one stops the run.

---

## Documentation and Media

| Resource | Link |
|---|---|
| Project Paper | [View paper](https://drive.google.com/file/d/1iKnfQ0rO-9pqQZX2WC6-bb9SaQvr3f34/view?usp=sharing) |
| Demo Video (Level 1) | [Watch demo](https://drive.google.com/file/d/1mv55oWKijQlvkETMPqagLJS9aK8EdErO/view?usp=sharing) |

---

## Overview

`ai_game.py` is the orchestrator. It runs the three levels in sequence, and each level must succeed before the next begins:

```
Level 1 (DQN)  ──pass──▶  Level 2 (Adversarial AI)  ──pass──▶  Level 3 (Pathfinding)
     │                          │                                    │
     └── fail ──▶ stop          └── fail ──▶ stop                    └── end
```

| | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| **Technique** | Deep Q-Network | Utility-driven FSM + A* | A* / Dijkstra |
| **Opponent** | Bullet, ghost, saws | Sabotaging enemy AI | Chasing adversary |
| **Learned?** | Yes — trained policy | No — scripted planning | No — classical search |
| **Success** | Reach gate with clue | Escape through door | Reach exit alive |

---

## Game Levels

### Level 1 — Hazard Escape (Deep RL)

A trained **DQN agent** must collect a randomly-placed clue, then reach the escape gate on the left wall — while dodging a travelling bullet, a ghost that actively chases it, and two saws patrolling vertically at 30% and 60% of screen width.

- **Environment** — custom `gymnasium.Env` ([`BulletDodgeEnv`](level1/final.py)) with a **22-feature normalized observation vector** and **5 discrete actions** (no-op, up, down, left, right).
- **Network** — 5-layer MLP: `22 → 1024 → 512 → 256 → 128 → 5`, with **LayerNorm**, **0.2 dropout**, and **Xavier initialization**.
- **Training** — **prioritized experience replay** (TD-error weighted sampling), a **target network** for stable bootstrapping, Adam with L2 weight decay, γ = 0.995, batch size 512, and ε-greedy exploration decaying 1.0 → 0.02.
- **Inference** — [`run.py`](level1/run.py) loads `best_model.pth` and acts **purely greedily** (`argmax` over Q-values, no exploration) in a 30 FPS loop. A reward ≥ 6000 marks the gate as reached.
- **Two-stage goal** — the gate stays closed until the clue is collected, forcing the policy to learn sequential objectives rather than a single beeline.

**Measured performance** (across 1,000 training episodes):

| Hazard | Avoidance rate |
|---|---|
| Bullet | 90% |
| Ghost | 80% |
| Saw | 87% |

### Level 2 — Lever Escape Challenge (Adversarial AI)

Fully autonomous **AI vs AI**. An `AgentAI` must set **four levers** to a randomly generated target combination to unlock the exit door, while an `EnemyAI` actively works against it.

- **Randomized puzzle** — the target sequence is regenerated each run, so the solution cannot be memorized.
- **Agent** ([`agent_ai.py`](The%20project%20itself/src/components/agent_ai.py)) — verifies lever states against the target, picks the next incorrect lever, navigates with **A\***, detects and recovers from stuck states, and breaks off to evade when threatened.
- **Adversary** ([`enemy_ai.py`](The%20project%20itself/src/components/enemy_ai.py)) — patrols the lever positions, **toggles correct levers back to sabotage progress**, and engages the agent in combat when in range.
- **Hazards** — three saws move along fixed paths and can damage and stun the agent.
- **Engine** — a small custom entity-component engine (`Entity`, `Body`, `Sprite`, `Trigger`) with tile maps, physics triggers, inventory, and combat.

The core challenge is the **contested objective**: progress is not monotonic, since the enemy can undo completed work, which requires the agent to continuously re-verify state rather than plan once.

### Level 3 — Maze Survival (Classical Search)

The agent crosses a **17 × 25 grid maze** seeded with **25 traps** and **35 fires**, starting at 100 health, while an adversary pursues it.

- **A\*** ([`level3.py`](AI%20Proj/level3.py)) with a hazard-weighted cost function that prioritizes safe routes over short ones.
- **Dijkstra** as an alternative planner, with a toggle for whether hazards may be traversed at all.
- **Adversary-avoidance pathfinding** — a modified A\* that adds a repulsion cost around the adversary's position, so the agent routes *away* from the threat rather than straight to the exit.
- **Consequences** — fire drains health, traps **stun** the agent for a set number of turns, and both agent and adversary leave visible trace trails for inspection.

---

## Setup

Requires **Python 3.10–3.13**. PyTorch does not yet publish wheels for 3.14.

```bash
git clone git@github.com:moneeb07/Escape-game-rl.git
cd Escape-game-rl
```

<details open>
<summary><b>Using uv (recommended)</b></summary>

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install pygame numpy gymnasium torch
```
</details>

<details>
<summary><b>Using standard venv + pip</b></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pygame numpy gymnasium torch
```
</details>

> **CPU-only machines:** install the smaller CPU build of PyTorch (≈190 MB instead of ≈2.5 GB):
> `uv pip install torch --torch-backend=cpu`

### Dependencies

| Package | Purpose |
|---|---|
| `torch` | DQN network and checkpoint loading (Level 1) |
| `gymnasium` | RL environment interface (Level 1) |
| `numpy` | Observation encoding and numeric ops |
| `pygame` | Rendering, audio, and input across all levels |

---

## Running

**Full three-level run:**

```bash
source .venv/bin/activate
python ai_game.py
```

**Individual levels:**

```bash
cd level1 && python run.py                      # Level 1 — DQN agent
cd "The project itself" && python src/game.py   # Level 2 — adversarial escape room
cd "AI Proj" && python level3.py                # Level 3 — maze survival
```

**Performance benchmark** — runs Level 1 repeatedly and reports a success rate:

```bash
python perfomance.py
```

> All levels open a real pygame window, so a desktop session is required. Over SSH you will need X forwarding.

---

## Project Structure

```
.
├── ai_game.py              # Orchestrator — runs all three levels in sequence
├── perfomance.py           # Success-rate benchmarking harness
├── level1/                 # Level 1 — Deep RL
│   ├── final.py            #   BulletDodgeEnv (Gymnasium) + DQN definition
│   ├── run.py              #   Inference entry point (greedy policy)
│   ├── train_check.py      #   Multi-episode evaluation
│   └── *.pth               #   Trained checkpoints (best_model.pth is the default)
├── The project itself/     # Level 2 — Adversarial escape room
│   ├── src/components/     #   agent_ai, enemy_ai, pathfinding (A*), puzzle, combat, saw
│   ├── src/core/           #   engine, area, map, camera, input
│   ├── src/stages/         #   menu, play (level setup and win condition)
│   └── content/            #   sprites, maps, fonts
└── AI Proj/
    └── level3.py           # Level 3 — maze, A*, Dijkstra, adversary avoidance
```

---

## Tech Stack

**Python** · **PyTorch** · **Gymnasium** · **NumPy** · **pygame**

**Techniques:** Deep Q-Learning · Prioritized experience replay · Target networks · Custom Gym environments · A\* and Dijkstra pathfinding · Adversarial agent design · Finite-state behaviour trees · Entity-component architecture

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `_pickle.UnpicklingError` / `WeightsUnpickler error` on load | PyTorch ≥ 2.6 defaults `torch.load` to `weights_only=True`. The loaders pass `weights_only=False` — verify yours does too. |
| `No module named 'torch'` | The virtual environment is not activated: `source .venv/bin/activate`. |
| `FileNotFoundError` on sprites or `.pth` files | Levels resolve assets relative to the working directory. Launch via `ai_game.py`, or `cd` into the level folder first. |
| `pygame.error: No available video device` | No display available. Run on a desktop session or enable X forwarding. |
