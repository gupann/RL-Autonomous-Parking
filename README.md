# Model-Based vs Model-Free Reinforcement Learning for Autonomous Parking

A comparative study of **Model-Based RL (Learned Dynamics + MPC)** and **Model-Free RL (SAC + HER)** on precision-critical autonomous parking tasks, focusing on **data efficiency** and **generalization** in constrained environments.

---

## Overview

Autonomous parking is a challenging control problem requiring:
- High positional accuracy
- Robust collision avoidance
- Reliable performance with limited data

While **model-free RL** methods can learn complex behaviors, they are often data-hungry and brittle in narrow, collision-sensitive settings. **Model-based RL** promises better sample efficiency and interpretability, but is rarely evaluated on tight-geometry parking tasks.

This project provides a **controlled, head-to-head comparison** between the two paradigms under identical environments, reward structures, and evaluation protocols.

---

## Research Questions

**RQ1 — Data Efficiency**  
How much data is required by MBRL (MPC with learned dynamics) vs. MFRL (SAC + HER) to achieve comparable parking performance?

**RQ2 — Generalization**  
Given a fixed training budget, which approach is more robust to *unseen* parking layouts and obstacle configurations?

---

## Environments

We implement **Gymnasium-style** parking environments inspired by HighwayEnv:

- **Parking Types**
  - Reverse Parking
  - Parallel Parking (novel implementation)

- **Obstacle Settings**
  - No obstacles
  - Static obstacles
  - Dynamic obstacles (moving vehicles)

- **Dynamics**
  - Kinematic bicycle model
  - Non-holonomic constraints
  - Actuation and sensor noise
  - Continuous control (throttle, steering)

---

## Methods

### Model-Based RL (MBRL)
- Learned vehicle dynamics model (MLP)
- Model Predictive Control (MPC) planner
- Gradient-based trajectory optimization
- Explicit collision avoidance
- Replans at every timestep

### Model-Free RL (MFRL)
- Soft Actor–Critic (SAC)
- Hindsight Experience Replay (HER)
- Goal-conditioned policy
- Sparse terminal rewards
- Large replay buffer and deep networks

---

## Key Results

### Data Efficiency (RQ1)
- MPC achieves **near-perfect success** with **~5k trajectories**
- SAC + HER requires **tens of thousands of interaction steps**
- MFRL data requirements grow rapidly with obstacle complexity

### Generalization (RQ2)
- MPC maintains **100% success** on unseen layouts
- SAC + HER success drops sharply under distribution shift
- Model-free policies tend to overfit training geometries

### Qualitative Behavior
- SAC + HER produces smoother trajectories
- MPC produces sharper corrective maneuvers but higher final accuracy
- Explicit planning improves safety in tight environments

---

## Dependencies

- Python 3.9+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- Stable-Baselines3
- HighwayEnv (adapted)

---
## Limitations & Future Work

- No curriculum learning across parking environments  
- No comfort-related metrics (e.g., jerk, smoothness) included in the reward function  
- MPC computational cost increases with planning horizon  
- No real-world deployment or sim-to-real transfer evaluation  

**Future work** includes curriculum training across environments, comfort-aware reward shaping, and learning unified policies that generalize across parking scenarios.

---

## Team

**Team 16**

- Ayush Agrawal  
- Anmol Gupta  
- Mehak Singal  
- Rachita Rajesh  
- Saharsh Goenka  

---

## References

- Zhang et al., *Reinforcement Learning-Based Motion Planning for Automatic Parking System*, IEEE Access, 2020  
- Farama Foundation, *HighwayEnv*  
- Additional references are included in the final project report
