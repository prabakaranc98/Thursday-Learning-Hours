# Slides Outline — TLH-2: Applied Life: Intelligence Meets Living Systems

> **19 slides · 60-minute session**
> Format: Foundations (15 min) → Frontiers (15 min) → Demo (20 min) → Discussion (10 min)
> Design: Ivory background (#f7f3ea), copper accent (#8f3e22), EB Garamond serif

---

## Slide 1 — Cover

**Title:** Applied Life: Intelligence Meets Living Systems
**Subtitle:** AI × Biology — Foundation Models, Protein Design, Perturbation Biology & Engineering Living Systems
**Session:** TLH-2 · Thursday Learning Hours
**Eyebrow:** Foundations · Frontiers · Frameworks

*Visual: DNA double helix schematic in minimal line art, copper on ivory*

---

## 🏛️ SECTION 1 — FOUNDATIONS (15 min, Slides 2–6)

---

## Slide 2 — Why Biology Now?

**Heading:** The AlphaFold Moment — and What Comes After

- AlphaFold 2 (2020): protein structure prediction — Nobel Prize 2024
- The shift: from "biology is too complex for AI" → "AI is the primary tool of biology"
- Where we are now: FMs for every biological scale; the era of *engineering* life with AI

*Visual: Timeline bar 2020–2026 with milestone markers (AF2, ESM-2, Evo 2, AIDO, IsoDDE)*

---

## Slide 3 — The Biological Hierarchy

**Heading:** DNA → RNA → Protein → Cell → Tissue: A Hierarchy of Information

Table:
| Scale | Data Type | ML Problem | Key Challenge |
|---|---|---|---|
| DNA | 3B bp sequence | Genomic FM | Long-range regulatory dependencies |
| RNA | Gene expression | Expression modeling | Cell-type specificity |
| Protein | AA sequence + 3D | Structure prediction | Sequence → fold → function |
| Cell | scRNA-seq profile | Cell state modeling | ~20,000 dimensional |
| Tissue | Spatial + expression | Spatial modeling | Emergent organization |

*Visual: Vertical cascade diagram showing each level with data type annotation*

---

## Slide 4 — Biology as a Machine Learning Problem

**Heading:** Why Transformers Work Naturally on Biology

- DNA, RNA, Protein: all sequences → language models transfer directly
- Evolution = pre-training signal: billions of years of selection pressure in sequence variation
- Gene expression = high-dimensional vector → set/attention models work
- The "text" of biology: codons, amino acids, gene names — all tokenizable

**Key insight:** The co-evolutionary structure of protein sequences encodes 3D contacts and functional constraints — the same way language syntax encodes semantic relationships.

---

## Slide 5 — The Perturbation Paradigm (CRL Bridge)

**Heading:** CRISPR is a *do*-Intervention on the Biological Causal Graph

```
Observational:   measure gene expression across cells (P(X))
Interventional:  knock out gene A with CRISPR (P(X | do(A=0)))
Counterfactual:  what would this cell look like if we had NOT knocked out A?
```

- Perturb-seq = CRISPR + scRNA-seq at scale → interventional data on biological causal graph
- This is the *exact* setting CRL identifiability theorems require
- Replogle (2022): 9,000+ gene knockouts × 2.5M cells = the ImageNet of perturbation biology

**The open frontier:** Can we prove that learned perturbation representations are causally identifiable?

---

## Slide 6 — The 6 Problem Spaces

**Heading:** The Applied Life Landscape

*Visual: 2×3 tile grid with icon, title, and one-line description for each:*
1. 🧬 Foundation Models for Biology
2. 🔬 Protein Design & Engineering
3. 🧫 Perturbation Biology & Drug Discovery
4. 🔭 Causal & Mechanistic Understanding
5. 🔩 Synthetic Biology & Bioengineering
6. 🧠 Computational Neuroscience

---

## 🔭 SECTION 2 — FRONTIERS (15 min, Slides 7–11)

---

## Slide 7 — ESM-2 & Protein Language Models

**Heading:** ESM-2: Evolutionary Scale Modeling

- 15B parameter masked LM trained on 250M UniRef90 sequences
- Learns contact maps, secondary structure, conservation, functional sites — *without ever seeing structures*
- ESM-3 (2024): multimodal — sequence + structure + function as tokens simultaneously
- ESMFold: comparable to AlphaFold2, 60× faster

**Key result:** The protein language model learns 3D structure as an *emergent* property of sequence co-evolution. No structural supervision needed.

*Visual: ESM-2 attention head visualization showing learned contact patterns*

---

## Slide 8 — Evo 2 & Genome-Scale Models

**Heading:** Evo 2 — Reading the Code of Life

- 9 trillion DNA base pairs · All domains of life · 1M token context
- Single-nucleotide resolution across bacterial, archaeal, and eukaryotic genomes
- Capabilities: mutation effect prediction anywhere in a genome, CRISPR guide RNA design, novel gene design

**What 1M context unlocks:** entire bacterial genomes; long-range regulatory elements; operon-level understanding

*Visual: Scale comparison — GPT-4 context vs. Evo 2 context vs. E. coli genome size*

---

## Slide 9 — AIDO: The Multi-Scale FM System

**Heading:** GenBio AI's AIDO — Biology's Hierarchy as a Model System

*Visual: Vertical architecture diagram*
```
AIDO.DNA  →  AIDO.RNA  →  AIDO.Protein  →  AIDO.Cell  →  AIDO.Tissue
                                                    ↕
                                        AIDO.StructureDiffusion
```

- SOTA on 300+ biological tasks across all scales
- First system of *connected* FMs reflecting biology's actual causal structure
- AIDO.Cell: predict cellular response to any perturbation — the "virtual cell"
- CRL connection: Eric Xing + Kun Zhang bridge paper (ICLR 2025)

**The open question:** Are AIDO's representations causally identifiable, or just predictively powerful?

---

## Slide 10 — Protein Design: From Prediction to Engineering

**Heading:** AlphaFold → RFdiffusion → ProteinMPNN: The Design Pipeline

```
Target Definition
      ↓
RFdiffusion (backbone design) → conditioned diffusion over 3D coordinates
      ↓
ProteinMPNN (sequence design) → inverse folding → millions of candidates
      ↓
AlphaFold 3 (structure validation) → self-consistency check
      ↓
ESM-2 fitness prediction → rank candidates on fitness landscape
      ↓
Wet lab synthesis of top-k candidates
```

**Case study:** de novo binder design — design a protein that binds a target receptor with no known binder. Previously took years; now takes days computationally.

---

## Slide 11 — Perturbation Prediction: GEARS

**Heading:** GEARS — Predicting Unseen Gene Perturbation Combinations

- Norman 2019 dataset: 131 genes × 131 genes = 17,000+ combinations. Wet lab measured ~2,000.
- Challenge: predict transcriptional response to *unseen* combinations from single-gene data
- GEARS approach: gene interaction knowledge graph (STRING) + GNN propagation

**Results:** GEARS can predict combinatorial perturbation effects for novel combinations not seen during training — including synergistic and antagonistic genetic interactions.

**Why this matters for drug discovery:** Most drug combinations are untested. AI can pre-screen in silico.

---

## 🛠️ SECTION 3 — FRAMEWORKS + DEMO (20 min, Slides 12–16)

---

## Slide 12 — The Bio ML Stack

**Heading:** Tools of the Trade

| Layer | Tools |
|---|---|
| **Protein FMs** | ESM-2/3, AlphaFold 3, RFdiffusion, ProteinMPNN |
| **Single-cell** | Scanpy, AnnData, scVI-tools, CellxGene Census |
| **Perturbation** | GEARS, scGen, CellOT, Pertpy |
| **Drug discovery** | DeepChem, RDKit, TDC, REINVENT 4 |
| **Datasets** | UniProt/PDB, CellxGene, LINCS L1000, ProteinGym |
| **Benchmarks** | FLIP, PEER, ProteinGym, CausalBench, DREAM5 |
| **Infrastructure** | Hugging Face (genbio-ai), CZ Science Virtual Cells |

---

## Slide 13 — Live Demo 1: ESM-2 Protein Embeddings

**Heading:** Demo 01 — Protein Language Model Embeddings & Fitness Landscape

**What we do:**
1. Load ESM-2 (650M) from Meta's pre-trained checkpoint
2. Embed a set of TP53 protein variants (missense mutations)
3. Visualize the embedding space — do similar sequences cluster?
4. Correlate embedding distances with functional fitness scores (ProteinGym data)
5. Zero-shot: use log-likelihood as a fitness predictor — no fine-tuning

**Key result to show:** ESM-2 log-likelihoods correlate with experimental fitness scores (AUROC ~0.75–0.85 on ProteinGym) without any task-specific training.

*Live notebook: `experiments/demo_01_esm2_protein_embeddings.ipynb`*

---

## Slide 14 — Live Demo 2: Single-Cell Perturbation Analysis

**Heading:** Demo 02 — scRNA-seq Analysis + Perturbation Response

**What we do:**
1. Load Norman 2019 dataset subset (10X Genomics, CRISPR Perturb-seq)
2. Preprocess with Scanpy: quality control, normalization, HVG selection
3. PCA → UMAP: visualize cell states in 2D
4. Compare gene expression profiles: control vs. perturbed cells
5. Identify differentially expressed genes post-perturbation
6. Basic latent shift: compute the "perturbation vector" in PCA space

**Key insight:** Even simple linear methods (PCA + nearest neighbor) can capture perturbation effects. This motivates why more sophisticated causal models (GEARS, scVI) are needed for unseen combinations.

*Live notebook: `experiments/demo_02_singlecell_perturbation.ipynb`*

---

## Slide 15 — Benchmark Landscape

**Heading:** How We Evaluate — Key Benchmarks in AI + Biology

| Benchmark | Task | Metric |
|---|---|---|
| **ProteinGym** | Fitness prediction across 250 proteins | Spearman ρ, AUROC |
| **FLIP** | Fitness landscape inference, 5 tasks | Spearman ρ |
| **PEER** | Molecular property prediction | RMSE, AUROC |
| **CausalBench** | Gene regulatory network inference from Perturb-seq | AUROC, EPR |
| **DREAM5** | Gene network inference benchmark | AUROC |
| **TDC** | Drug discovery tasks (ADMET, binding, generation) | Multiple |
| **sci-Plex / Norman 2019** | Perturbation response prediction | Pearson r, MSE |

**The evaluation challenge:** Task performance ≠ biological understanding. A model can ace ProteinGym without understanding *why* mutations affect fitness.

---

## Slide 16 — Landscape Map + Key Players

**Heading:** The Applied Life Ecosystem

*Visual: 2-axis map — (Molecular ←→ Cellular) × (Prediction ←→ Design)*

Positioned:
- DeepMind/Isomorphic: Molecular, Design (protein + drug)
- Arc Institute: Molecular, Prediction (genome-scale FM)
- GenBio AI: Full spectrum (AIDO spans all scales)
- Recursion: Cellular, Design (phenotypic drug discovery)
- CZ Science: Cellular, Prediction/Infrastructure (virtual cells)
- CMU-CLeaR: Foundational theory (CRL for all domains)

---

## 🔄 SECTION 4 — DISCUSSION (10 min, Slides 17–19)

---

## Slide 17 — Open Problems

**Heading:** The Deep Open Questions

1. **Evaluation**: How do we measure biological *understanding* vs. pattern memorization?
2. **Causal faithfulness**: Are bio FM representations causally identifiable? What test would prove it?
3. **Scale vs. structure**: Do bigger models (Evo 2, AIDO) automatically learn more causally faithful representations?
4. **Interventional sufficiency**: Does Perturb-seq satisfy the intervention conditions for CRL identifiability?
5. **Cross-scale integration**: Can a single model genuinely learn across DNA, protein, and cell scales, or does the hierarchy require explicit grounding?

---

## Slide 18 — Connections to Other Pillars

**Heading:** Applied Life × The PraCha Labs Ecosystem

| Pillar | Connection |
|---|---|
| **Applied Causality** | CRL identifiability for bio FM representations; treatment effect estimation from Perturb-seq |
| **Applied Worlds** | World models for cellular simulation; SCM-structured bio models |
| **Applied Cogs** | AI lab assistants; cognitive architectures for scientific discovery |
| **Applied Models** | Continual learning for evolving biological data; fine-tuning bio FMs |
| **Applied Algorithms** | Evolutionary algorithms for protein design; Bayesian optimization for molecular search |

---

## Slide 19 — Key Takeaways + Next Session

**Heading:** What We Covered + Where We Go Next

**Key takeaways:**
- We are in the foundation model era for biology — the field is moving in months
- Every bio FM predicts well but is causally opaque — identifiability is the open frontier
- CRISPR + Perturb-seq provides real interventional data — CRL theory applies
- The AIDO system is the most architecturally faithful to biological reality
- Perturbation prediction is the problem with the clearest CRL connection

**Next session options (vote):**
- TLH-3A: Deep dive on CRL × Perturbation Biology — identifiability from Perturb-seq data
- TLH-3B: Deep dive on Protein Design — RFdiffusion + ProteinMPNN hands-on
- TLH-3C: Deep dive on AIDO.Cell — virtual cell modeling and evaluation

---

## Design Notes

- **Color:** ivory background `#f7f3ea`, copper accent `#8f3e22`, charcoal text `#17130e`
- **Typography:** EB Garamond for headings, IBM Plex Mono for code/data labels
- **Diagrams:** thin-stroke line art, no gradients; architecture boxes in pale cream `#ede8d8` with copper borders
- **Data viz:** Seaborn `muted` palette, minimal gridlines
- **Slide format:** 16:9, left-aligned text blocks, max 5 bullet points per slide
- **Section dividers:** copper horizontal rule with section name in mono uppercase
