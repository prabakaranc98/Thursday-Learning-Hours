# Reading List — TLH-1: Applied Cogs × RL × LLMs

> Organised by theme. Papers marked ⭐ are essential reading before the session.

---

## 🧠 Cognitive Science Foundations

| # | Paper / Book | Year | Key Contribution | Notes |
|---|---|---|---|---|
| 1 | ⭐ Anderson et al. — *An Integrated Theory of the Mind* (ACT-R 6.0) | 2004 | Unified cognitive architecture with memory modules | Read §2–§4 |
| 2 | Baars — *A Cognitive Theory of Consciousness* | 1988 | Global Workspace Theory | Book; skim Ch. 1–3 |
| 3 | ⭐ Clark — *Surfing Uncertainty* | 2016 | Predictive processing / free energy | Book intro + Ch. 1 |
| 4 | Friston — *The free-energy principle: a unified brain theory?* | 2010 | Active inference formalism | Nature Reviews Neuroscience |
| 5 | Laird — *The Soar Cognitive Architecture* | 2012 | Production-rule system, problem spaces | MIT Press |
| 6 | Miller, Galanter & Pribram — *Plans and the Structure of Behavior* | 1960 | Goal/sub-goal hierarchy; TOTE units | Classic — skim |

---

## 🤖 LLM Agents & Reasoning

| # | Paper | Year | Key Contribution | Notes |
|---|---|---|---|---|
| 7 | ⭐ Yao et al. — *ReAct: Synergizing Reasoning and Acting* | 2023 | Interleaved thinking + action baseline | ICLR 2023 |
| 8 | Wei et al. — *Chain-of-Thought Prompting* | 2022 | Eliciting step-by-step reasoning | NeurIPS 2022 |
| 9 | Shinn et al. — *Reflexion: Language Agents with Verbal RL* | 2023 | Self-reflective agents using verbal feedback | NeurIPS 2023 |
| 10 | Park et al. — *Generative Agents* | 2023 | Believable agents with memory + reflection | CHI 2023 |
| 11 | ⭐ Wang et al. — *A Survey on Large Language Model-based Autonomous Agents* | 2023 | Comprehensive survey | arXiv:2308.11432 |
| 12 | Sumers et al. — *Cognitive Architectures for Language Agents* | 2024 | Connects cog-sci to LLM agents | TMLR |

---

## 🎮 Reinforcement Learning

| # | Paper | Year | Key Contribution | Notes |
|---|---|---|---|---|
| 13 | ⭐ Christiano et al. — *Deep RL from Human Preferences* | 2017 | RLHF foundation | NeurIPS 2017 |
| 14 | Bai et al. — *Constitutional AI: Harmlessness from AI Feedback* | 2022 | RLAIF | arXiv:2212.08073 |
| 15 | Rafailov et al. — *Direct Preference Optimisation* | 2023 | DPO — simpler than PPO | NeurIPS 2023 |
| 16 | ⭐ Shao et al. — *DeepSeekMath (GRPO)* | 2024 | Group Relative Policy Optimisation | arXiv:2402.03300 |
| 17 | Ng, Harada & Russell — *Policy Invariance Under Reward Transformations* | 1999 | Potential-based reward shaping proof | ICML 1999 |
| 18 | Sutton & Barto — *Reinforcement Learning: An Introduction* | 2018 | Canonical textbook | Free online |
| 19 | ⭐ Hafner et al. — *Mastering Diverse Domains through World Models* | 2023 | Dreamer-V3 | arXiv:2301.04104 |

---

## 🔗 RL + LLMs (Post-Training & Agents)

| # | Paper | Year | Key Contribution | Notes |
|---|---|---|---|---|
| 20 | ⭐ Ouyang et al. — *Training language models to follow instructions (InstructGPT)* | 2022 | RLHF applied to LLMs | NeurIPS 2022 |
| 21 | Ziegler et al. — *Fine-Tuning Language Models from Human Preferences* | 2019 | Early RLHF for text | arXiv:1909.08593 |
| 22 | Carta et al. — *Grounding Large Language Models in Interactive Environments with Online RL* | 2023 | Online RL fine-tuning of LLMs | ICML 2023 |
| 23 | Kwon et al. — *Reward Design with Language Models* | 2023 | LLM-generated reward functions | ICLR 2023 |
| 24 | Chen et al. — *Decision Transformer* | 2021 | RL as sequence modelling | NeurIPS 2021 |
| 25 | ⭐ Hu et al. — *LoRA: Low-Rank Adaptation* | 2022 | Efficient fine-tuning for RL updates | ICLR 2022 |

---

## 🏢 Enterprise AI & Agentic Systems

| # | Paper / Article | Year | Key Contribution | Notes |
|---|---|---|---|---|
| 26 | ⭐ Shen et al. — *HuggingGPT: Solving AI Tasks with ChatGPT* | 2023 | LLM as task planner over AI models | NeurIPS 2023 |
| 27 | Significant Gravitas — *AutoGPT* | 2023 | Early autonomous GPT-4 agent | GitHub |
| 28 | ⭐ Microsoft Research — *TaskBench: Benchmarking LLMs in Task Automation* | 2023 | Enterprise task evaluation | arXiv:2311.18760 |
| 29 | Chase — *LangChain Docs: Agents* | 2023 | Production agent patterns | docs.langchain.com |
| 30 | Wu et al. — *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation* | 2023 | Multi-agent orchestration | arXiv:2308.08155 |

---

## 📚 Suggested Reading Order

**Absolute minimum (before session):**
1. Wang et al. (2023) survey — #11
2. Yao et al. ReAct — #7
3. Christiano et al. RLHF — #13
4. Shao et al. GRPO — #16
5. Hafner et al. Dreamer-V3 — #19

**Deeper dive (after session):**
- ACT-R paper (#1) + Cognitive Architectures for Language Agents (#12)
- Ng et al. reward shaping (#17)
- Full Sutton & Barto textbook (#18)

---

## 📂 Paper PDFs

Save downloaded PDFs into a `reading/papers/` subfolder (gitignored for size).
Add a one-sentence summary as a comment next to each entry above when you read it.
