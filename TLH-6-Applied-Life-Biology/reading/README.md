# Reading List — TLH-2: Applied Life: Intelligence Meets Living Systems

> 30 papers across 5 themes. ⭐ = essential pre-session reading (read at minimum these 6).
> Suggested order: Theme 1 → Theme 3 → Theme 4 → Theme 2 → Theme 5.

---

## Theme 1: Biological Foundation Models

*Large-scale pre-trained models learning biological representations.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 1 | **ESM-2: Language Models of Protein Sequences at Scale** | Lin et al. (Meta AI) | 2023 | The foundational protein LM — understand how evolutionary co-variation encodes structure/function |
| ⭐ 2 | **AIDO: A Foundation Model System Spanning the Molecular and Cellular Hierarchy of Life** | Xing et al. (GenBio AI) | 2025 | The most ambitious bio FM system — architecture, philosophy, 300+ task evaluation |
| 3 | **Evo 2: DNA Foundation Models from Genomes to Molecules** | Nguyen et al. (Arc Institute) | 2024 | Genome-scale FM, 9T bp, 1M context — understand the scaling regime |
| 4 | **Geneformer: Transfer Learning Enables Predictions in Network Biology** | Theodoris et al. (Gladstone) | 2023 | Single-cell FM using rank-value encoding of gene expression |
| 5 | **scGPT: Toward Building a Foundation Model for Single-Cell Multi-omics** | Cui et al. (UofT) | 2024 | Generative pre-training for single-cell, zero-shot cell type annotation |
| 6 | **Nucleotide Transformer: Building Robust Foundation Models for Human Genomics** | Dalla-Torre et al. (Bioptimus) | 2023 | DNA FM with comprehensive benchmark evaluation, understanding genomic sequences |

---

## Theme 2: Protein Structure & Design

*From structure prediction to programmable protein engineering.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 7 | **Highly Accurate Protein Structure Prediction with AlphaFold** | Jumper et al. (DeepMind) | 2021 | The breakthrough — Nobel Prize 2024. Understand Evoformer + pair representation |
| 8 | **De Novo Design of Proteins with RFdiffusion** | Watson et al. (Baker Lab) | 2023 | Diffusion over protein backbone coordinates — the generative design breakthrough |
| 9 | **Robust Deep Learning–Based Protein Sequence Design Using ProteinMPNN** | Dauparas et al. (Baker Lab) | 2022 | Inverse folding — given a backbone, design the sequence |
| 10 | **Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3** | Evans et al. (DeepMind) | 2024 | Extends AF2 to all biomolecules — proteins, DNA, RNA, ligands, ions |
| 11 | **Accurate Prediction of Protein Structures Using RoseTTAFold** | Baek et al. (Baker Lab) | 2021 | Parallel to AF2 — three-track network, important for understanding the field |
| 12 | **ESM-3: Simulating 500 Million Years of Evolution with a Language Model** | Hayes et al. (EvolutionaryScale) | 2024 | Multimodal — sequence + structure + function as tokens. Generative design. |

---

## Theme 3: Perturbation Biology & Drug Discovery

*Predicting cellular responses to interventions at genome scale.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 13 | **Exploring Genetic Interaction Manifolds Constructed from Rich Single-Cell Phenotypes** | Norman et al. | 2019 | The foundational combinatorial Perturb-seq dataset — the benchmark everyone uses |
| ⭐ 14 | **GEARS: Predicting Transcriptional Outcomes of Novel Multi-Gene Perturbations** | Roohani et al. | 2023 | State-of-the-art perturbation prediction — knowledge graph + GNN |
| 15 | **Mapping Information-Rich Genotype-Phenotype Landscapes with Genome-Scale Perturb-seq** | Replogle et al. | 2022 | 9,000+ knockouts, 2.5M cells — the ImageNet of perturbation biology |
| 16 | **scGen Predicts Single-Cell Perturbation Responses** | Lotfollahi et al. | 2019 | VAE-based perturbation prediction — latent shift approach |
| 17 | **Optimal Transport for Single-Cell Genomics (CellOT)** | Bunne et al. | 2023 | OT framing of perturbation — maps unperturbed → perturbed cell distributions |
| 18 | **Identifying Genes and Cells Responsible for Perturbation Response (CINEMA-OT)** | Tian et al. | 2024 | Causal-inspired: separates shared vs. treatment-specific effects |

---

## Theme 4: Causal & Mechanistic Understanding

*Moving beyond prediction to provably faithful biological representations.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 19 | **Toward Causal Representation Learning** | Schölkopf et al. | 2021 | The foundational CRL survey — the theoretical lens for everything in bio |
| 20 | **Causal Representation Learning from Multimodal Biomedical Observations** | Sun, Zhang, Xing et al. | ICLR 2025 | **THE bridge paper** — CRL identifiability for biological data (Kun Zhang + Eric Xing) |
| 21 | **Deep Generative Modeling for Single-Cell Transcriptomics (scVI)** | Lopez et al. | 2018 | Foundational VAE for single-cell — the baseline every paper builds on |
| 22 | **CausCell: Causal Disentanglement for Single-Cell Transcriptomics** | — | 2025 | First causal disentanglement + counterfactual generation for single-cell |
| 23 | **scCausalVI: Causality-Aware Generative Model for Perturbation Responses** | — | 2025 | Structural causal modeling applied to single-cell perturbation |
| 24 | **CausalBench: A Large-scale Benchmark for Network Inference from Single-cell Perturbation Data** | Chevalley et al. | 2023 | Benchmark for recovering gene regulatory networks from perturbation data |

---

## Theme 5: Drug Discovery & Synthetic Biology

*AI closing the design-build-test-learn loop in therapeutic and synthetic contexts.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| 25 | **A Deep Learning Approach to Antibiotic Discovery** | Stokes et al. | 2020 | Landmark: AI discovers halicin, a novel antibiotic — the proof of concept |
| 26 | **Therapeutics Data Commons: Machine Learning Datasets and Tasks for Therapeutics** | Huang et al. | 2022 | The standard benchmark suite for drug discovery ML |
| 27 | **Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules** | Gómez-Bombarelli et al. | 2018 | Generative molecular design via VAE — foundational molecular generation |
| 28 | **Model-Based Reinforcement Learning for Biological Sequence Design** | Angermueller et al. | 2019 | RL for sequence design — closes the active learning loop in protein/DNA engineering |
| 29 | **REINVENT 4: Modern AI-Driven Generative Molecule Design** | Loeffler et al. | 2024 | Production-grade generative chemistry for drug design |
| 30 | **Large Language Models for Synthetic Biology** | Inukai et al. | 2024 | LLMs for gene circuit design — the frontier of synthetic biology AI |

---

## Suggested Reading Order

**Before the session (⭐ essentials, ~8 hrs):**
1. **ESM-2** (#1) — 45 min read — understand protein LMs
2. **AIDO** (#2) — 90 min read — understand the full bio FM vision
3. **AlphaFold 2** (#7) — 60 min read — understand the structure breakthrough
4. **Norman et al. 2019** (#13) — 45 min read — understand Perturb-seq
5. **GEARS** (#14) — 60 min read — current SOTA for perturbation prediction
6. **Schölkopf et al. 2021** (#19) — 60 min read — the CRL lens

**To go deeper:**
- CRL × Bio bridge: read #20 (Sun et al. ICLR 2025) immediately after #19
- Protein design pipeline: read #8 (RFdiffusion) + #9 (ProteinMPNN) as a pair
- Perturbation models: read #16 (scGen) → #17 (CellOT) → #14 (GEARS) in order

**For the drug discovery track:**
- Start with #25 (Stokes 2020) for inspiration, then #26 (TDC) for benchmarks
