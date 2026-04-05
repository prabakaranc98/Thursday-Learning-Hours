# Experiments — TLH-1: Applied Cogs × RL × LLMs

## Contents

| File | Type | Description |
|---|---|---|
| `demo_01_cognitive_rl_agent.ipynb` | Notebook | Cognitive memory module + RL policy walk-through |
| `demo_02_rl_llm_tool_use.py` | Script | RL-trained tool-selection policy for multi-step tasks |
| `requirements.txt` | Config | Python dependencies |

## Setup

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Set API keys (if using OpenAI / Anthropic)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Launch notebooks
jupyter notebook demo_01_cognitive_rl_agent.ipynb
```

## Experiment Goals

1. **Demo 01** — Show that a cognitively-inspired memory module (episodic + semantic) outperforms a vanilla RAG baseline on a multi-step reasoning task.
2. **Demo 02** — Train a lightweight RL policy (PPO via TRL) on tool-selection decisions and compare to greedy/ReAct baselines.

## Results Log

| Experiment | Baseline | Cognitive+RL | Delta |
|---|---|---|---|
| Multi-step QA (HotpotQA-50) | TBD | TBD | TBD |
| Tool selection accuracy | TBD | TBD | TBD |

*Populate after running experiments.*
