# TLH-2 · Applied Life: Intelligence Meets Living Systems

> **Thursday Learning Hours — Session 2**
> *AI × Biology — Foundation Models, Protein Design, Perturbation Biology & Engineering Living Systems*

---

## 🧬 Motivation

The question isn't just "can AI predict biology?" — it's **"can AI *understand* biology well enough to engineer it?"**

AlphaFold won a Nobel Prize. Evo 2 reads 9 trillion base pairs across all domains of life with a 1M-token context window. GenBio AI's AIDO spans the entire biological hierarchy from nucleotide to cell. We are firmly in the **foundation model era for biology** — and the questions have shifted.

Biology is the richest, most complex domain for AI: hierarchical (DNA → RNA → Protein → Cell → Tissue), interventional (CRISPR is a real *do*-operation), multi-scale, and genuinely causal. This session maps the full landscape — where AI meets living systems, who is building what, and where the deep open problems live.

This is a **survey session** — TLH-2 opens the Applied Life pillar. Future sessions will go deep on specific problem spaces.

---

## 🗺️ Problem Spaces

| # | Problem Space | Core Question | Key Methods |
|---|---|---|---|
| 1 | **Foundation Models for Biology** | Can a single FM learn the language of life across scales? | ESM-3, AIDO, Geneformer, Evo 2 |
| 2 | **Protein Design & Engineering** | Can we design proteins with desired functions de novo? | AlphaFold 3, RFdiffusion, ProteinMPNN |
| 3 | **Perturbation Biology & Drug Discovery** | How does a cell respond to interventions at scale? | GEARS, scGen, CellOT, Perturb-seq |
| 4 | **Causal & Mechanistic Understanding** | Do bio FMs actually *understand* biology? | CRL, SCMs, CausalBench, CausCell |
| 5 | **Synthetic Biology & Bioengineering** | Can AI close the design-build-test-learn loop? | mRNA design, gene circuit AI |
| 6 | **Computational Neuroscience** | What can AI teach us about how the brain computes? | Neural decoding, connectomics |

---

## 🗂️ Folder Structure

```
TLH-2-Applied-Life-Biology/
│
├── README.md                              ← you are here
│
├── tex/                                   ← LaTeX technical report
│   └── main.tex
│
├── slides/                                ← Presentation deck outline
│   └── SLIDES_OUTLINE.md
│
├── notes.md                               ← Deep session notes & architecture diagrams
├── resources.md                           ← Curated tools, frameworks, datasets, benchmarks
├── links.md                               ← 80+ curated reference links
│
├── reading/                               ← 30-paper annotated reading list
│   └── README.md
│
└── experiments/                           ← Demo notebooks & scripts
    ├── README.md
    ├── demo_01_esm2_protein_embeddings.ipynb
    ├── demo_02_singlecell_perturbation.ipynb
    └── requirements.txt
```

---

## 🎯 Core Thesis

> *"Biology is the ultimate complex system. Foundation models have transformed what's possible — but they lack causal guarantees. The open frontier is not just prediction, but understanding: representations that provably correspond to real biological mechanisms. Biology provides the interventional data CRL theory needs. AI provides the scale biology needs. The marriage of the two is the next frontier."*

---

## 🔬 3F Structure (Foundations · Frontiers · Frameworks)

| Pillar | TLH-2 Coverage |
|---|---|
| 🏛️ **Foundations** | Central dogma as a data problem; sequence → structure → function paradigm; biological scales and their data types; perturbation as causal intervention |
| 🔭 **Frontiers** | AIDO multi-scale FM; Evo 2 genome-scale FM; AlphaFold 3; RFdiffusion + ProteinMPNN; CRL × single-cell; virtual cell models |
| 🛠️ **Frameworks** | ESM-2, Scanpy/AnnData, CellxGene Census, GEARS, DeepChem, RDKit; hands-on protein embeddings + single-cell demo |

---

## 🏢 Landscape — Key Players

| Company / Lab | Focus | Key Contribution |
|---|---|---|
| **GenBio AI** | Multi-scale bio FMs (AIDO) | First connected FM system spanning all biological scales — DNA through tissue |
| **DeepMind / Isomorphic Labs** | Protein structure & drug design | AlphaFold (Nobel Prize 2024), IsoDDE drug design engine |
| **Arc Institute** | Genome-scale FMs (Evo 2) | 9T base pairs, all domains of life, 1M-token context |
| **Recursion** | AI-driven drug discovery | Largest proprietary cellular imaging dataset |
| **Bioptimus** | Universal bio FM | Multi-modal, multi-scale foundation model |
| **CZ Science** | Virtual cells platform | Open infrastructure for bio FMs (hosts AIDO.Cell) |
| **Genentech / Roche** | FM + agents for drug design | Pharma-scale AI integration |
| **CMU-CLeaR (Kun Zhang)** | CRL theory for biology | Causal identifiability for biological FM representations |

---

## 🚀 Quick Start

```bash
# Install experiment dependencies
pip install -r experiments/requirements.txt

# Demo 1: ESM-2 protein embeddings + zero-shot fitness prediction
jupyter notebook experiments/demo_01_esm2_protein_embeddings.ipynb

# Demo 2: Single-cell RNA-seq analysis + perturbation response
jupyter notebook experiments/demo_02_singlecell_perturbation.ipynb
```

> **Note:** Demo 1 runs on CPU/any GPU. Demo 2 requires ~8GB RAM for the single-cell dataset.
> For full AIDO models (AIDO.Cell), an A100 GPU is recommended.

---

## 📅 Session Info

- **Date:** TBD
- **Format:** 60-min seminar (15 min Foundations + 15 min Frontiers + 20 min Live Demo + 10 min Discussion)
- **Type:** Landscape Survey — maps the Applied Life pillar; future sessions drill deeper
- **Audience:** AI/ML researchers and engineers curious about biology applications
- **Pre-reads:** Start with `reading/README.md` — prioritize Theme 1 (Bio FMs) and Theme 3 (Perturbation Biology)

---

*Part of the [Thursday Learning Hours](../README.md) initiative — Applied Life pillar.*
