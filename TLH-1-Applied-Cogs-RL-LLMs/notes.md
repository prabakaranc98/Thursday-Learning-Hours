# Session Notes — TLH-1: Applied Cognitive Science × RL × LLMs

> **Date:** TBD  
> **Duration:** 60 minutes  
> **Format:** Theory (20 min) → Live Demo (20 min) → Discussion (20 min)

---

## 🧠 Core Thesis

> *"Scaffolding is a starting point, not a destination. To build agents that genuinely plan, learn, and adapt in enterprise environments, we must go beyond prompt engineering and look to Cognitive Science and Reinforcement Learning."*

---

## Section 1 — Why Scaffolding Isn't Enough

### Key Points

- Most production LLM agents today are **reactive** — they respond to a context window, not a persistent internal state.
- Scaffolding frameworks (LangChain, LlamaIndex, AutoGen) are powerful orchestration tools but they don't provide:
  - Reward signals or optimisation loops
  - Principled memory management
  - Uncertainty-aware decision-making
  - Long-horizon credit assignment

### Critical Failure Modes Observed

| Failure Mode | Root Cause | Cognitive/RL Fix |
|---|---|---|
| Tool hallucination | No grounding in task state | RL policy learns when to use tools |
| Context overflow | No memory management | Episodic + semantic memory modules |
| Reward hacking | Proxy metrics, not true goals | Careful reward shaping + RLAIF |
| Repetitive loops | No metacognitive monitoring | Confidence gate / stopping criterion |
| Catastrophic forgetting | Stateless sessions | Persistent episodic memory |

---

## Section 2 — Cognitive Science Insights

### ACT-R Takeaways for Agents
- **Procedural memory** → tool/skill library (functions the agent can call)
- **Declarative memory** → episodic + semantic stores (retrieved by activation)
- **Spreading activation** → approximate with cosine similarity in a vector DB

### Global Workspace Theory → Attention
- The LLM context window *is* the global workspace — information that enters context gets "broadcast" to all layers.
- Design implication: **selective context curation** > raw RAG dumping.
- Only the most relevant, high-salience chunks should enter the workspace at each step.

### Predictive Processing → Metacognition
- The brain is a prediction machine; prediction errors drive learning.
- For agents: model your own uncertainty.
- Implementation: calibrated output probabilities, entropy-based routing, ask-for-help triggers.

---

## Section 3 — RL Concepts Applied to Agents

### RLHF → RLAIF → GRPO Timeline
```
2017  RLHF (OpenAI / InstructGPT)
2022  Constitutional AI / RLAIF (Anthropic)
2023  DPO — Direct Preference Optimisation
2024  GRPO — Group Relative Policy Optimisation (DeepSeek)
2025  RLVR — RL with Verifiable Rewards (reasoning models)
```

### Reward Design Principles
1. **Dense rewards** beat sparse — instrument intermediate milestones.
2. **Potential-based shaping** is safe — preserves optimal policy.
3. **Avoid proxy metrics** — reward task completion, not token count.
4. **Human-in-the-loop checkpoints** for high-stakes actions.

### Hierarchical RL — Why It Matters
- Sub-goal decomposition mirrors how humans break complex tasks into steps.
- High-level planner sets the goal; low-level executor handles micro-decisions.
- In practice: planner LLM + executor specialised model/agent.

### World Models in Practice
- Dreamer-V3 trains a compact latent world model — the agent imagines rollouts before acting.
- For enterprise: simulate what happens if the agent sends an email, modifies a DB record, etc. before committing.
- Trade-off: expensive to train, but saves costly real-world mistakes.

---

## Section 4 — Architecture Proposal

```
┌─────────────────────────────────────────────────────────┐
│                     LLM Backbone                        │
│              (GPT-4o / Llama-3 / Mistral)               │
└────────────┬──────────────────────────┬─────────────────┘
             │                          │
     ┌───────▼──────┐          ┌────────▼───────┐
     │  Episodic    │          │   Semantic      │
     │  Memory      │          │   Knowledge     │
     │  (Vector DB) │          │   Graph         │
     └───────┬──────┘          └────────┬────────┘
             │                          │
     ┌───────▼──────────────────────────▼────────┐
     │           Context Curator (GWT)            │
     │     Selective broadcast into context       │
     └────────────────────┬──────────────────────┘
                          │
               ┌──────────▼──────────┐
               │  RL Policy Head     │
               │  (tool selection +  │
               │   action ordering)  │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │  Metacognitive Gate │
               │  (confidence check) │
               └──────────┬──────────┘
                          │
                   Act / Query / Escalate
```

---

## 💡 Key Takeaways

1. Cognitive architectures provide **structural blueprints** for agent components.
2. RL provides the **optimisation loop** that makes agents improve over time.
3. The combination unlocks agents that are **adaptive, principled, and auditable**.
4. Enterprise deployment requires thinking about **cost of mistakes** — world models and metacognition address this.

---

## ❓ Discussion Questions

- What reward signals are realistically available in your enterprise setting?
- How do you handle catastrophic actions (irreversible tool use) in RL training?
- Is the cognitive architecture analogy helpful, or does it mislead engineering decisions?
- What is the right level of human oversight in a cognitively-inspired RL agent?

---

## 📝 Action Items from Session

- [ ] Run experiments in `experiments/` and populate results table in `tex/main.tex`
- [ ] Collect enterprise use-case data for reward shaping study
- [ ] Survey latest RLVR (RL with Verifiable Rewards) papers
- [ ] Draft TLH-2 topic based on discussion outcomes
