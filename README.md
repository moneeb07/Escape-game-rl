🧠 RL Escape Room with Adversarial AI

An AI-powered escape room game designed to demonstrate the practical application of Deep Reinforcement Learning, adversarial AI, intelligent puzzles, and autonomous navigation. The project features a three-level environment where an agent must survive hazards, solve puzzles, and outmaneuver enemies to escape.

🚀 Overview

This project explores how intelligent agents can learn, adapt, and make decisions in dynamic and hostile environments. Using PyTorch and Gymnasium, the system integrates reinforcement learning with classical AI techniques such as A* pathfinding to create realistic gameplay scenarios.

🎮 Game Levels
🔹 Level 1 — Hazard Escape

The agent operates in a high-risk arena where it must dodge bullets, avoid moving threats, and retrieve a clue required for progression.

Performance Stats:

✅ 90% Bullet Avoidance

✅ 80% Ghost Evasion

✅ 87% Saw Avoidance
(Measured across 1,000 training episodes)

🔹 Level 2 — Lever Escape Challenge

The agent must activate four levers in the correct sequence to unlock the exit while an adversarial enemy sabotages progress. Movement is further complicated by saw hazards that can damage and stun the agent.

🔹 Level 3 — Maze Survival

The agent navigates a complex maze filled with fire hazards, traps, and stun mechanisms while avoiding enemy interference. Success depends on selecting safe routes and adapting to environmental threats.

🧩 Key Features

Deep Q-Learning agent trained through reward-based learning

Custom Gymnasium environment with stochastic interactions

Adversarial enemy behaviors

Intelligent puzzle mechanics

Hazard-aware A* pathfinding

State-driven decision making

Memory-based puzzle verification

🛠️ Tech Stack

Python

PyTorch

Gymnasium

NumPy
