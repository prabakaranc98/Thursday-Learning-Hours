# Resources — TLH-1: Applied Cogs × RL × LLMs

> Curated tools, repositories, datasets, and frameworks relevant to this session.

---

## 🛠️ Frameworks & Libraries

### Agent Frameworks
| Resource | Description | Link |
|---|---|---|
| LangGraph | Stateful, graph-based agent orchestration | https://github.com/langchain-ai/langgraph |
| AutoGen | Multi-agent conversation framework (Microsoft) | https://github.com/microsoft/autogen |
| OpenAI Agents SDK | Lightweight production agent framework | https://github.com/openai/openai-agents-python |
| Haystack | Production NLP pipelines with agent support | https://github.com/deepset-ai/haystack |

### RL Libraries
| Resource | Description | Link |
|---|---|---|
| Stable-Baselines3 | Reliable RL algorithm implementations | https://github.com/DLR-RM/stable-baselines3 |
| TRL (HuggingFace) | Transformer RL — PPO, DPO, GRPO | https://github.com/huggingface/trl |
| RLlib (Ray) | Scalable distributed RL | https://docs.ray.io/en/latest/rllib/ |
| Dreamer-V3 | World model implementation | https://github.com/danijar/dreamerv3 |
| VerifAI / VPO | RL with verifiable rewards | TBD |

### Cognitive Architecture Tools
| Resource | Description | Link |
|---|---|---|
| ACT-R Python | Python interface for ACT-R | https://github.com/asmaloney/ACT-R |
| MemGPT | LLM with paged memory management | https://github.com/cpacker/MemGPT |
| Zep | Long-term memory for LLM agents | https://github.com/getzep/zep |
| LangMem | Memory-augmented LangChain agents | https://github.com/langchain-ai/langmem |

### Evaluation & Benchmarks
| Resource | Description | Link |
|---|---|---|
| ToolBench | Benchmark for LLM tool use | https://github.com/OpenBMB/ToolBench |
| SWE-bench | Software engineering agent benchmark | https://www.swebench.com/ |
| AgentBench | Multi-task agent evaluation | https://github.com/THUDM/AgentBench |
| GAIA | General AI Assistants benchmark | https://huggingface.co/datasets/gaia-benchmark/GAIA |

---

## 📦 Datasets

| Dataset | Domain | Use |
|---|---|---|
| HotpotQA | Multi-hop QA | Long-chain reasoning evaluation |
| WebArena | Web navigation | Tool-use in realistic browser env |
| ALFWorld | Household tasks | Embodied agent + RL |
| MiniWoB++ | Web form tasks | GUI tool-use baseline |
| D4RL | Offline RL | Pre-collected trajectory datasets |

---

## 🔧 Experiment Stack (this sprint)

```
Python       3.11+
torch        2.3+
transformers 4.41+
trl          0.9+
openai       1.30+
langchain    0.2+
langgraph    0.1+
faiss-cpu    1.8+
networkx     3.3+
gymnasium    0.29+
stable-baselines3  2.3+
jupyter      latest
```

Install:
```bash
pip install -r experiments/requirements.txt
```

---

## 🌐 Online Courses & Tutorials

| Resource | Platform | Link |
|---|---|---|
| Spinning Up in Deep RL | OpenAI | https://spinningup.openai.com |
| Deep RL Course | HuggingFace | https://huggingface.co/learn/deep-rl-course |
| CS285: Deep RL | UC Berkeley | https://rail.eecs.berkeley.edu/deeprlcourse/ |
| Cognitive Science MOOC | Coursera (Duke) | https://www.coursera.org/learn/cognitive-neuroscience |

---

## 🎙️ Talks & Videos

- **Yann LeCun** — "A Path Towards Autonomous Machine Intelligence" (Meta AI, 2022)
- **Pieter Abbeel** — "Deep Reinforcement Learning" (ICML 2021 Tutorial)
- **Karl Friston** — "Active Inference and the Free Energy Principle"
- **Andrej Karpathy** — "The Unreasonable Effectiveness of Recurrent Neural Networks" (still relevant for sequence agents)
- **Sam Altman / Ilya Sutskever** — discussions on scaling and agency (various interviews)

---

## 🗣️ Communities

- [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/)
- [EleutherAI Discord](https://discord.gg/eleutherai) — open research
- [Alignment Forum](https://www.alignmentforum.org/) — safety + cognition
- [Papers With Code — Agents](https://paperswithcode.com/task/autonomous-agents)
