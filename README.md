# Adaptive Entropy-Normalized Proximal Policy Optimization (AEN-PPO)
### Reinforcement Learning in Dynamic Action Spaces for Open-Pit Mining Haul Truck Dispatching

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29%2B-green.svg)](https://gymnasium.farama.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.5%2B-yellow.svg)](https://www.pygame.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📌 Overview

This repository provides the complete implementation of **Adaptive Entropy-Normalized Proximal Policy Optimization (AEN-PPO)** applied to multi-truck dispatching in open-pit mining operations.

In constrained, safety-critical industrial environments, the set of feasible actions dynamically changes from state to state:
* Haul trucks cannot be dispatched if they are already in transit or loading.
* Shovels and excavation pits become blocked when their physical queue capacity is exhausted.
* Haul corridors enforce speed limits, one-way directional flow, and collision avoidance.

### The Problem
* **Standard PPO** relies on negative reward penalties to learn feasibility. In complex combinatorial dispatching, random exploration almost never hits valid actions ($87.8\%$ invalid action rate), resulting in severe policy collapse and trial-and-error damage.
* **Standard Masked PPO** removes invalid actions using logit masking, ensuring feasibility. However, it suffers from **Exploration Starvation**: when the number of valid choices shrinks (e.g. from 12 to 2), the raw Shannon entropy $H(\pi)$ drops proportionally to $\ln(|A_{\text{valid}}|)$. This artificially suppresses the exploration bonus by over $70\%$, causing premature convergence, greedy policy lock-in, and severe shovel queue bunching.

### The Solution: AEN-PPO
**AEN-PPO** couples state-dependent action masking with a **logarithmically normalized policy entropy bonus**:
$$H_{\text{norm}}(\pi(\cdot \mid s)) = \begin{cases} \dfrac{H(\pi(\cdot \mid s))}{\ln(|A_{\text{valid}}(s)|)}, & \text{if } |A_{\text{valid}}(s)| > 1 \\ 0, & \text{if } |A_{\text{valid}}(s)| \le 1 \end{cases}$$

This guarantees scale-invariant exploration $\in [0, 1]$ across all states, regardless of whether 2 or 12 actions are physically valid.

---

## 🔬 Mathematical Formulation

### 1. Action Masking Layer
Invalid logits are masked to $-\infty$ (or float minimum) prior to softmax normalization:
$$\tilde{z}_a = \begin{cases} z_a, & \text{if } M_a(s) = 1 \\ -\infty, & \text{if } M_a(s) = 0 \end{cases}$$
$$\pi_\theta(a \mid s) = \frac{\exp(\tilde{z}_a)}{\sum_{a' \in A_{\text{valid}}(s)} \exp(\tilde{z}_{a'})}$$

### 2. Clipped Surrogate Objective with Normalized Entropy
The total optimization loss minimized during backpropagation is:
$$\mathcal{L}_{\text{total}}(\theta, \phi) = L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\phi) - c_2 H_{\text{norm}}(\pi_\theta)$$

where:
* $L^{\text{CLIP}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$
* $L^{\text{VF}}(\phi) = \hat{\mathbb{E}}_t \left[ \left( V_\phi(s_t) - \hat{R}_t \right)^2 \right]$
* $H_{\text{norm}}(\pi_\theta) = \hat{\mathbb{E}}_t \left[ \frac{H(\pi(\cdot \mid s_t))}{\ln(|A_{\text{valid}}(s_t)|)} \right]$

---

## 📊 Key Experimental Results

Evaluated across **225 experimental runs** ($15 \text{ configurations} \times 3 \text{ algorithms} \times 5 \text{ training seeds} \times 10 \text{ evaluation seeds}$):

| Evaluation Dimension | Standard PPO | Masked PPO | **Proposed AEN-PPO** | AEN-PPO Advantage |
| :--- | :---: | :---: | :---: | :---: |
| **Invalid Action Rate** | $87.84\%$ | $0.00\%$ | **$0.00\%$** | $100\%$ Feasible Enforcement |
| **Operational Success Rate** | $0.0\%$ | $100.0\%$ | **$100.0\%$** | Zero constraint violations |
| **Convergence Speed ($N^*$)** | $>50$ Ep (Failed) | $4.4 \pm 3.2$ Ep | **$4.4 \pm 3.2$ Ep** | **$11.4\times$ faster convergence** |
| **Cumulative Regret ($R_N$)** | $23,286.1$ | $1,525.8$ | **$1,506.5$** | **$15.4\times$ lower regret** |
| **4-Pit Performance ($P=4$)** | $-545.49$ | $-137.45$ | **$-133.37$** | **$p = 0.015$ superiority (t = 2.52)** |
| **Shovel Queue Waiting Time** | $9.75\text{ min}$ | $9.39\text{ min}$ | **$8.76\text{ min}$** | **$7\%\text{--}9\%$ less truck idling** |
| **Generalization Gap ($\Delta_{\text{gen}}$)** | $1.00$ (Failed) | $8.36$ | **$6.98$** | **$16.5\%$ smaller transfer gap** |
| **Inference Latency** | $0.087\text{ ms}$ | $0.104\text{ ms}$ | **$0.105\text{ ms}$** | Real-time capable ($<1$ ms) |

---

## 🏗️ Repository Architecture

```text
AEN-PPO/
├── agents/                         # RL Agent Implementations
│   ├── ppo.py                      # Standard PPO Baseline
│   ├── masked_ppo.py               # Masked PPO Agent (Action Masking)
│   └── aen_ppo.py                  # Proposed Adaptive Entropy-Normalized PPO
├── environments/                   # Mine Simulation & Grid Mechanics
│   ├── grid_map.py                 # 2D Grid Topology & Tile Definitions
│   ├── visual_truck_dispatch_env.py# Primary Scalable Discrete-Event Haulage Env
│   ├── truck_dispatch_env.py       # Abstract Discrete-Event Benchmark Env
│   ├── action_mask.py              # Physical Constraint & Feasibility Rules
│   └── pathfinding.py              # A* / Dijkstra Haul Road Routing
├── visualization/                  # Pygame Graphical Interface
│   ├── simulator_window.py         # Real-time Visual Simulation Window
│   ├── controls.py                 # UI Buttons & Interactive Controls
│   └── map_editor.py               # Interactive Mine Map Designer GUI
├── maps/                           # Mine Grid Topologies (JSON)
│   ├── starter_map.json            # 24x16 Baseline Mine Map
│   ├── pit_1_map.json              # Single-Pit Mine
│   ├── pit_2_map.json              # 2-Pit Standard Mine
│   ├── pit_3_map.json              # 3-Pit Mine
│   ├── pit_4_map.json              # 4-Pit Complex Multi-Shovel Mine
│   └── evaluation_map.json         # Unseen Generalization Test Map
├── training/                       # Policy Training Pipelines
│   ├── train_aen_ppo.py            # Train AEN-PPO
│   ├── train_masked_ppo.py         # Train Masked PPO
│   ├── train_ppo.py                # Train Standard PPO
│   └── train_visual.py             # Train on Visual Grid Maps
├── experiments/                    # Comprehensive Benchmark Suite
│   └── run_comprehensive_benchmarks.py # 225-Run Multi-Seed Scalability Benchmarks
├── evaluation/                     # Metric Computations & Publication Plotting
│   ├── metrics.py                  # 10 Operational & RL Evaluation Metrics
│   ├── generate_plots.py           # Publication Plots 01 to 13
│   ├── generate_strong_claim_visuals.py # Plots 14 to 17 (Significance & Transfer)
│   └── generate_regret_convergence_success_plots.py # Plots 18 to 21
├── results/                        # Evaluation Tables & Checkpoints
│   ├── evaluation/plots/           # 21 Publication-Quality Plots (300 DPI)
│   ├── evaluation/tables/          # Markdown and CSV Benchmark Metrics
│   └── evaluation/raw_results/     # Raw Trajectory Logs Across All Seeds
├── tests/                          # Pytest Unit & Regression Suite
│   ├── test_action_masks.py        # Mask Correctness Tests
│   ├── test_grid_map.py            # Map Geometry Tests
│   ├── test_metrics.py             # Metric Calculation Tests
│   ├── test_pathfinding.py         # Routing Engine Tests
│   └── test_scalability.py         # Environment Scaling Tests
├── requirements.txt                # Python Dependencies
├── pytest.ini                      # Pytest Configuration
└── run_visual_simulation.py        # Master GUI Launcher
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/pavirb475/RL_Adaptive-Entropy-Normalized-Action-Masking-for-Reinforcement-Learning-in-Dynamic-Action-Spaces.git
cd RL_Adaptive-Entropy-Normalized-Action-Masking-for-Reinforcement-Learning-in-Dynamic-Action-Spaces
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 1. Interactive 2D Visual Simulation
Watch haul trucks navigate roads, queue at shovels, deliver ore, and get dispatched by trained agents in real time:

```bash
# Run with Proposed AEN-PPO
python run_visual_simulation.py --agent aen_ppo --trucks 4 --jobs 6 --map maps/pit_2_map.json

# Run on Complex 4-Pit Multi-Shovel Mine
python run_visual_simulation.py --agent aen_ppo --trucks 6 --jobs 8 --map maps/pit_4_map.json

# Run on Unseen Generalization Map
python run_visual_simulation.py --agent aen_ppo --map maps/evaluation_map.json
```

**Simulation Keyboard Controls:**
* `[SPACE]` : Play / Pause
* `[RIGHT ARROW]` : Step 1 simulation step
* `[UP] / [DOWN]` : Increase / Decrease simulation speed
* `[A]` : Cycle between RL agents live (`AEN-PPO` $\rightarrow$ `Masked PPO` $\rightarrow$ `PPO` $\rightarrow$ `Random`)
* `[P]` : Toggle planned route trajectories
* `[R]` : Reset episode
* `[ESC]` : Exit window

---

### 2. Interactive Mine Map Editor
Draw and design your own open-pit mine layouts with a click-and-drag GUI:
```bash
python run_visual_simulation.py --mode editor --map maps/my_custom_mine.json
```
* **Tools:** Paint Haul Roads, Pit Walls, Shovel Loading Stations, Dump Crushers, and Truck Depots.
* **Validation:** Built-in A* path connectivity validation ensures all loading and dumping stations are reachable.

---

### 3. Training Reinforcement Learning Agents
Train agents from scratch on the mine dispatching environment:

```bash
# Train Proposed AEN-PPO
python training/train_aen_ppo.py --episodes 50 --trucks 4 --jobs 6

# Train Masked PPO
python training/train_masked_ppo.py --episodes 50 --trucks 4 --jobs 6

# Train Standard PPO
python training/train_ppo.py --episodes 50 --trucks 4 --jobs 6
```

---

### 4. Running Comprehensive Benchmarks (225 Experimental Runs)
Execute the multi-seed evaluation across all 15 configurations ($N \in \{2, 4, 6, 8\}$, $P \in \{1, 2, 3, 4\}$, $J \in \{4, 6, 8, 12\}$, 5 training seeds, 10 evaluation seeds):
```bash
python experiments/run_comprehensive_benchmarks.py
```

---

### 5. Generating Publication Plots & Tables
Regenerate all 21 publication figures (300 DPI) and Markdown summary tables:
```bash
# Plots 01 to 13 (Learning curves, throughput, scalability, radar chart)
python evaluation/generate_plots.py

# Plots 14 to 17 (Multi-pit superiority, entropy mechanism, transfer slopegraph)
python evaluation/generate_strong_claim_visuals.py

# Plots 18 to 21 (Cumulative regret, convergence speed, success rate)
python evaluation/generate_regret_convergence_success_plots.py
```

---

### 6. Running Unit & Regression Tests
Verify that environment mechanics, action masking, pathfinding, and evaluation metrics are 100% functional:
```bash
pytest tests/
```
*(All 28 tests pass in $<2$ seconds)*

---

## 📜 Dependencies

* Python 3.10+
* PyTorch (`torch >= 2.0.0`)
* Gymnasium (`gymnasium >= 0.29.0`)
* Pygame (`pygame >= 2.5.0`)
* NumPy (`numpy >= 1.24.0`)
* Pandas (`pandas >= 2.0.0`)
* Matplotlib (`matplotlib >= 3.7.0`)
* SciPy (`scipy >= 1.10.0`)
* Tabulate (`tabulate >= 0.9.0`)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
