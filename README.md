
# **Secure Multi-Agent Pong Environment (PPO Reinforcement Learning)**

*A PyTorch + Pygame + RL project with independent PPO agents*

## 🚀 Overview

This project implements a **real-time AI-vs-AI Pong environment**, where **two independent PPO (Proximal Policy Optimization) agents** compete against each other.
The environment simulates full RL gameplay, complete with a training pipeline, agent statistics, and secure model checkpoints.

---

## 🧠 Key Features

### 🎮 **1. AI vs AI Reinforcement Learning**

* Two PPO agents trained independently
* Continuous interaction between agents in a custom Pong environment
* Real-time match visualization using **Pygame**

### 🏋️ **2. Complete Training Pipeline**

* 100+ episodes per agent
* Observations, actions, rewards, rollout collection, optimization steps
* Supports early stopping, checkpointing, and evaluation loops

### 📊 **3. Detailed Performance Statistics**

For each agent, the following stats are tracked:

* Average score
* Best & worst score
* Win rates across episodes
* Score differentials

Displayed after every training session.

### 🔐 **4. Secure Model Saving / Loading**

* Models saved with **SHA-256 integrity hashing**
* Hash verification on load ensures:

  * No corruption
  * No tampering
  * Full reliability in experiments

---

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **Pygame**
* **NumPy**
* **Hashlib (SHA-256)**

---


