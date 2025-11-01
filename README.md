# Q-Learning – Discount Factor Experiments (CartPole-v1)

This repository contains the code used to run a series of controlled experiments on how the **discount factor γ (gamma)** affects learning and convergence in **Q-Learning**, using the **CartPole-v1** environment from Gymnasium.

---

## 🎯 Goal

The purpose of this project is to **demonstrate, visualize, and compare** how different discount factors (`γ`) influence the learning stability, speed, and convergence of a Q-Learning agent in the CartPole environment.

Each run stores TensorBoard logs and Q-tables that can be used to analyze the relationship between immediate vs. long-term reward optimization.

---

## 🧩 Features

- Modular and clean architecture (each component in a dedicated file)  
- Centralized hyperparameter configuration via `config.py`  
- TensorBoard logging for episode rewards and convergence  
- Deterministic runs through fixed random seeds  
- Supports multiple gamma values: `{0.100, 0.900, 0.950, 0.990, 0.995, 0.998, 1.100}`  

---

## ⚙️ Installation & Requirements

To run this experiment, you need **Python 3.10+** and the following libraries:

- **Gymnasium**  
- **NumPy**  
- **TensorBoard**

If you haven’t installed Gymnasium yet, follow this detailed step-by-step tutorial on my website:  
👉 **[How to Install OpenAI Gymnasium in Windows and Launch Your First Python RL Environment](https://www.reinforcementlearningpath.com/how-to-install-openai-gymnasium-in-windows-and-launch-your-first-python-rl-environment/)**

## How to Run
### Train the Agent	

```bash
python main.py
```


### Run Demo Mode (Playback)

```bash
python main.py --demo
```