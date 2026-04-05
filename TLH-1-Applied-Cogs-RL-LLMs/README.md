# TLH-1 · Applied Cognitive Science × RL × LLMs

> **Thursday Learning Hours — Session 1**
> *How Cognitive Science and Reinforcement Learning can solve enterprise-level AI/Agentic problems — beyond simple scaffolding and harness.*

---

## 🎯 Motivation

Modern LLM-based agents are powerful but brittle. Most enterprise deployments rely on prompt engineering, retrieval augmentation, and thin orchestration layers (scaffolding). While these work for narrow tasks, they fail under:

- Long-horizon planning with delayed rewards
- Dynamic, partially observable environments
- Adaptive decision-making under uncertainty
- Multi-step tool use with strategic reasoning

**Cognitive Science** and **Reinforcement Learning** offer principled frameworks to address these gaps — grounding agent design in how intelligent systems (biological and artificial) actually learn, plan, and act.

---

## 🗂️ Folder Structure

```
TLH-1-Applied-Cogs-RL-LLMs/
│
├── README.md               ← you are here
│
├── tex/                    ← LaTeX report / technical paper
│   └── main.tex
│
├── slides/                 ← Presentation deck (PPTX/PDF)
│   └── SLIDES_OUTLINE.md
│
├── notes.md                ← Session notes & key insights
├── resources.md            ← Curated tools, repos, datasets
│
├── reading/                ← Reading list & paper summaries
│   └── README.md
│
└── experiments/            ← Demo notebooks & Python scripts
    ├── README.md
    ├── demo_01_cognitive_rl_agent.ipynb
    └── demo_02_rl_llm_tool_use.py
```

---

## 🧠 Core Themes

| Theme | Description |
|---|---|
| **Cognitive Architectures** | ACT-R, Global Workspace Theory, predictive processing applied to LLM agents |
| **RL for Agentic Reasoning** | RLHF, RLAIF, GRPO, reward shaping for multi-step tool use |
| **Memory & Working Memory** | Episodic/semantic memory modules inspired by cognitive science |
| **Planning Under Uncertainty** | Model-based RL, World Models, and their enterprise applications |
| **Meta-Learning & Adaptation** | Few-shot and continual learning strategies for adaptive agents |

---

## 🚀 Quick Start

```bash
# Install experiment dependencies
pip install -r experiments/requirements.txt

# Launch the demo notebook
jupyter notebook experiments/demo_01_cognitive_rl_agent.ipynb
```

---

## 📅 Session Info

- **Date:** TBD
- **Format:** 60-min seminar (20 min theory + 20 min live demo + 20 min discussion)
- **Audience:** AI/ML practitioners, researchers, and engineers

---

*Part of the [Thursday Learning Hours](../README.md) initiative.*
