# Slides Outline — TLH-1: Applied Cogs × RL × LLMs

> **File:** `TLH-1-Applied-Cogs-RL-LLMs.pptx` *(build from this outline)*
> **Tool recommendation:** Use [Slidev](https://sli.dev/) for code-friendly slides, or export this outline to PowerPoint / Google Slides.

---

## Deck Structure (60-min session)

### Cover Slide
- Title: *Applied Cognitive Science × RL × LLMs*
- Subtitle: *Beyond Scaffolding — Principled Enterprise AI Agents*
- Session: TLH-1 | Thursday Learning Hours

---

### Section 1 — The Problem (5 min)

**Slide 1 · The scaffolding trap**
- Diagram: typical LangChain / AutoGen harness
- Pain points: brittleness, reward-free, memoryless

**Slide 2 · What "enterprise-grade" actually means**
- Long-horizon tasks, multi-turn, dynamic environments
- Real cost of failure: financial, reputational, compliance

---

### Section 2 — Cognitive Science Primer (10 min)

**Slide 3 · Cognitive architectures at a glance**
- ACT-R, SOAR, GWT, Predictive Processing — one-liner each
- Key insight: modularity + memory + metacognition

**Slide 4 · Global Workspace Theory → LLM context**
- Diagram: broadcast mechanism ↔ attention heads
- Selective broadcast idea

**Slide 5 · Memory types and agent design**
- Episodic (vector DB), Semantic (knowledge graph), Procedural (skills/tools)
- Map to concrete engineering components

**Slide 6 · Metacognition & uncertainty**
- "Knowing what you don't know" — calibrated confidence
- Practical: confidence head, human-in-the-loop triggers

---

### Section 3 — RL Fundamentals for Agents (10 min)

**Slide 7 · MDP refresher**
- $\mathcal{M} = \langle S, A, P, R, \gamma \rangle$
- Why this matters for tool-use agents

**Slide 8 · From RLHF → RLAIF → GRPO**
- Timeline diagram
- Trade-offs: human cost vs. scalability

**Slide 9 · Reward shaping for tool use**
- Potential-based shaping formula
- Example: intermediate milestones in a multi-step workflow

**Slide 10 · Hierarchical RL**
- High-level planner + low-level executor
- Maps to sub-agent patterns in enterprise AI

**Slide 11 · World Models (Model-Based RL)**
- Plan in imagination before acting
- Dreamer-V3 brief overview

---

### Section 4 — Putting It Together (5 min)

**Slide 12 · Proposed agent architecture**
- Diagram: LLM backbone + episodic memory + RL fine-tuning loop + metacognitive gate

**Slide 13 · Enterprise use cases**
- Customer support automation (long-horizon resolution)
- Code review / PR agent (tool-use + reasoning)
- Financial analysis agent (world model + uncertainty)

---

### Section 5 — Live Demo (20 min)

**Slide 14 · Demo overview**
- `demo_01_cognitive_rl_agent.ipynb` — cognitive memory module walkthrough
- `demo_02_rl_llm_tool_use.py` — RL-trained tool selection

**Slide 15 · Results preview**
- Baseline vs. cognitive+RL agent performance table
- Key take-away numbers

---

### Section 6 — Discussion & Q&A (10 min)

**Slide 16 · Open problems**
- Reward design, credit assignment, distribution shift

**Slide 17 · Research directions**
- Multi-agent cognitive roles, continual RL, spiking neural networks

**Slide 18 · Resources & reading**
- QR code → `resources.md` and `reading/README.md`

**Slide 19 · Thank you**
- GitHub repo link, contact info

---

## Design Notes

- **Colour scheme:** deep navy + accent amber (cognitive/warm tone)
- **Font:** Inter or Source Sans Pro for body; monospace for code snippets
- **Diagrams:** prefer simple boxes-and-arrows over dense flowcharts
- **Animations:** minimal — one build per complex diagram
