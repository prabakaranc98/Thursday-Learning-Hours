# Session Notes — TLH-2: Applied Life: Intelligence Meets Living Systems

> **Date:** TBD
> **Duration:** 60 minutes
> **Format:** Foundations (15 min) → Frontiers (15 min) → Demo (20 min) → Discussion (10 min)
> **Type:** Survey / Landscape Map

---

## 🧬 Core Thesis

> *"Biology is the ultimate complex system — hierarchical, interventional, multi-scale, and causally structured. Foundation models have cracked open the field: AlphaFold predicts structure, Evo 2 reads genomes, AIDO connects the whole hierarchy. But every model that predicts well is still causally opaque. The next frontier isn't better prediction — it's provably causal representations. And biology provides exactly the interventional data that CRL theory needs."*

---

## 🏛️ FOUNDATIONS

### Section 1 — The Biological Hierarchy as a Data Problem

Biology is not a flat domain — it is a **hierarchy of information** where each level transforms and regulates the one above it.

```
┌─────────────────────────────────────────────────────────────┐
│                    The Biological Hierarchy                  │
│                                                              │
│  DNA (genome)    → 3B base pairs per human cell             │
│       ↓            A/T/C/G alphabet, 4-base language        │
│  RNA (transcriptome) → ~20,000 genes expressed              │
│       ↓            mRNA, ncRNA, regulatory RNA              │
│  Protein (proteome) → ~100,000 unique proteins              │
│       ↓            20 amino acid alphabet, folds into 3D    │
│  Cell (single-cell) → ~37 trillion cells in the body        │
│       ↓            gene expression profile = cell state     │
│  Tissue / Organism → emergent behavior, disease, aging      │
└─────────────────────────────────────────────────────────────┘
```

**The Central Dogma as an ML problem:**
- DNA is a sequence → language model
- RNA expression is a vector over genes → gene expression modeling
- Protein is a sequence that folds into 3D → sequence + structure prediction
- Cell state is a high-dimensional vector (gene × cell matrix) → single-cell modeling

**Why this hierarchy is special for AI:**
1. Every level has a natural sequence/token structure → transformers work naturally
2. Evolution is nature's pre-training signal — billions of years of selection pressure
3. Biology is genuinely interventional — CRISPR lets us perform *do*-operations on the causal graph
4. Scale: billions of protein sequences, millions of single-cell profiles, full human genome

---

### Section 2 — Data Types at Each Scale

| Scale | Data Type | Key Technology | Size |
|---|---|---|---|
| **DNA** | Nucleotide sequences | Next-gen sequencing (NGS) | 3B bp per genome |
| **RNA** | Gene expression profiles | RNA-seq, scRNA-seq | ~20,000 genes per cell |
| **Protein** | Amino acid sequences + 3D structure | Cryo-EM, X-ray crystallography | 250M+ known sequences |
| **Cell** | Single-cell transcriptomics | 10X Genomics Chromium | Millions of cells per atlas |
| **Tissue** | Spatial transcriptomics | Visium, MERFISH | Spatial + expression maps |

**Key insight:** At every scale, the data is a **sequence or vector** — which is exactly what transformer architectures handle well. This is why language models transfer so cleanly to biology.

---

### Section 3 — The Perturbation Paradigm

> *Perturbation biology provides what CRL theory requires: real interventions on the causal graph of a complex system.*

**What is Perturb-seq?**
- Combines CRISPR knockouts (or drug treatments) with single-cell RNA sequencing
- You knock out a gene in a cell, then read out the full transcriptomic response
- At scale: Norman et al. 2019 knocked out combinations of genes; Replogle 2022 did genome-scale Perturb-seq

**Why perturbations are special:**
- CRISPR knockout = a genuine *do*-intervention: do(gene_X = 0)
- This is not a natural experiment or observational association — it is a physical intervention
- Pearl's do-calculus applies directly
- This gives us the data to recover causal structure — which is exactly what CRL identifiability theorems require

**The computational challenge:**
- ~20,000 genes × combinatorial knockouts = intractable wet lab exploration
- AI goal: predict the response to *unseen* perturbation combinations
- Generalize across cell types, organisms, experimental conditions

---

## 🔭 FRONTIERS

### Section 4 — Foundation Models for Biology

#### 4.1 ESM: The Protein Language Model (Meta AI)

ESM-2 (Lin et al., 2023) is the foundational protein language model:
- Trained on 250M UniRef90 protein sequences
- Architecture: BERT-style masked language model
- Sizes: 8M → 650M → 3B → 15B parameters
- **Key insight:** evolutionary co-variation signals in sequences → structural/functional representations

**What ESM-2 learns:**
- Contact maps (which amino acids are spatially close)
- Secondary structure (alpha helices, beta sheets)
- Conservation patterns across species
- Functional site annotations

ESMFold extends ESM-2 to structural prediction — comparable to AlphaFold2 but orders of magnitude faster.

ESM-3 (2024): multimodal — takes sequence, structure, AND function as input/output. Can generate proteins satisfying multiple constraints simultaneously.

#### 4.2 Evo 2: Genome-Scale Foundation Model (Arc Institute)

```
Evo 2 Stats:
  Training data: 9 trillion base pairs
  Organisms: bacteria, archaea, eukaryotes (all domains of life)
  Context window: 1,000,000 tokens (single-nucleotide resolution)
  Parameters: 7B
  Key capability: genome-scale generation and understanding
```

What Evo 2 can do:
- Predict the effect of any mutation anywhere in a genome
- Generate functional CRISPR guide RNAs
- Design novel protein-coding sequences
- Understand regulatory elements, promoters, enhancers

The 1M-token context is crucial — it lets the model see an entire bacterial genome or large eukaryotic gene locus at once. This captures long-range regulatory dependencies that smaller context models miss.

#### 4.3 AIDO: The Multi-Scale FM System (GenBio AI)

```
AIDO System Architecture (GenBio AI, Xing et al. 2025):

  AIDO.DNA ──────────────────────────────────────┐
  (7B params, DNA sequences)                      │
       ↓                                          │
  AIDO.RNA ──────────────────────────────────────┤
  (mRNA structure, expression)                    │
       ↓                                          ├── Connected via
  AIDO.Protein ──────────────────────────────────┤    hierarchical
  (sequence → structure → function)              │    biology
       ↓                                          │
  AIDO.Cell ─────────────────────────────────────┤
  (single-cell gene expression profiles)          │
       ↓                                          │
  AIDO.Tissue ────────────────────────────────────┘
  (spatial transcriptomics)
       ↓
  AIDO.StructureDiffusion
  (3D molecular structure generation)
```

**The key insight:** Biology is hierarchical by nature. Most bio FMs (ESM, Geneformer, Evo 2) are siloed at one scale. AIDO is the first *system* of connected FMs that mirrors the actual causal structure of biology — bottom-up, each level informing the next.

**AIDO.Cell achievements:**
- SOTA on 300+ single-cell tasks
- Virtual cell simulation: predict how a cell responds to a new drug or gene knockout
- Hosted on CZ Science Virtual Cells Platform

**The CRL connection:**
- Eric Xing (GenBio AI founder) co-authored with Kun Zhang (CMU-CLeaR): "CRL from Multimodal Biomedical Observations" (ICLR 2025)
- The strategic insight: AIDO learns *representations* at each scale, but there's no guarantee they are *causally identifiable*
- Open question: do AIDO's latent representations of cell state correspond to true biological causal factors?

---

### Section 5 — Protein Design: From Prediction to Engineering

The AlphaFold arc represents the fastest transition from "impossible" to "solved" in the history of ML:

```
2020  AlphaFold 2 — protein structure prediction (CASP14, near-experimental accuracy)
2021  RoseTTAFold — Baker Lab's parallel solution
2022  ProteinMPNN — inverse folding (given 3D backbone, design the sequence)
2022  RFdiffusion — diffusion over protein backbone coordinates for de novo design
2023  AlphaFold 2 database — 200M protein structures freely available
2024  AlphaFold 3 — predicts structure + interactions: proteins, DNA, RNA, ligands, ions
2026  IsoDDE (Isomorphic Labs) — doubles AF3 accuracy, predicts binding affinities
```

**The full protein design pipeline:**

```
  Target Definition         What do I want the protein to do?
        ↓                   (bind a specific receptor, catalyze a reaction)
  Backbone Design           RFdiffusion: diffuse over backbone coordinates
        ↓                   conditioned on desired function/binding site
  Sequence Design           ProteinMPNN: given backbone → design sequence
        ↓                   (inverse folding, millions of candidates)
  Structure Validation      AlphaFold 3: verify the designed sequence folds
        ↓                   as intended (self-consistency check)
  Fitness Prediction        ESM-2 / ProteinGym: score mutations on fitness landscape
        ↓
  Wet Lab Synthesis         Synthesize top candidates, test in vitro
```

**The fitness landscape problem:**
- A protein's function lives on a high-dimensional landscape over sequence space
- Directed evolution explores this landscape: random mutation → selection → repeat
- AI-guided: model the landscape, navigate it with gradient-based or evolutionary search
- Key insight: most proteins are in low-fitness regions; the functional space is sparse

---

### Section 6 — Perturbation Biology: Deep Dive

#### The Genome-Scale View

Replogle et al. (2022) — Genome-scale Perturb-seq:
- 2.5 million cells
- 9,000+ gene knockouts
- Full transcriptional response for every knockout

This dataset is the ImageNet of perturbation biology. It provides:
1. A map of genetic interactions across the entire genome
2. Ground truth for evaluating perturbation prediction models
3. The interventional signal CRL theory needs

#### Key Models for Perturbation Prediction

**scGen (Lotfollahi et al., 2019):**
- VAE encodes cells into a latent space
- Perturbation = a vector shift in latent space
- Works for seen perturbations; struggles with unseen combinations

**GEARS (Roohani et al., 2023):**
- Uses a gene interaction knowledge graph (STRING database)
- GNN learns to propagate perturbation effects through the graph
- Can predict combinatorial (multi-gene) perturbation effects
- State-of-the-art on Norman 2019 combinatorial benchmark

**CellOT (Bunne et al., 2023):**
- Frames perturbation as an optimal transport problem
- Learns the mapping from unperturbed → perturbed cell distribution
- Transport cost = biological plausibility

**The CRL opportunity:**
- All models above learn *correlational* mappings
- None prove that the learned perturbation direction corresponds to the *true causal mechanism*
- CausCell (2025) and scCausalVI (2025) begin to address this
- The open problem: identifiable causal factors from perturbation data with formal guarantees

---

### Section 7 — The CRL × Biology Bridge

```
The Universal Causal Structure:

  Domain        Latent Factors          Intervention          Observation
  ──────────    ───────────────         ─────────────         ──────────────
  Single-cell   Cell state (z_bio)      Gene knockout         Gene expression
                Batch effects (z_batch) Drug treatment        scRNA-seq profile
                Perturbation (z_pert)   (CRISPR / small mol)  (n_cells × n_genes)
  
  Protein       Structure (z_struct)    Mutation              Amino acid sequence
                Function (z_func)       Directed evolution    Fitness score
  
  Genomics      Regulatory programs     eQTL, GWAS            Genome sequence
                (z_reg)                 Natural variation      Expression levels
```

**Key theorem (informal):** If you observe a system under multiple interventions (gene knockouts), and the interventions shift sparse subsets of causal mechanisms independently, then the latent causal factors are identifiable from the observational/interventional data.

**Why this matters for biology:**
- Perturb-seq provides interventions that shift *one gene at a time* (sparse, targeted)
- This is exactly the multi-environment / multi-intervention setting that CRL identifiability results require
- The bridge paper: Sun et al. (ICLR 2025) — "CRL from Multimodal Biomedical Observations" (Kun Zhang + Eric Xing)

**Current state of the art:**
- CausCell (2025): causal disentanglement for single-cell, counterfactual prediction
- scCausalVI (2025): structural causal modeling for perturbation responses
- SENA-δ (de la Fuente, 2025): biologically interpretable latent causal factors
- scVI/sVAE+: sparse mechanism shift in single-cell latent space

**Open problem (potential research direction):**
> Prove that Perturb-seq data satisfies the interventional sufficiency conditions for CRL identifiability of bio FM representations. Design an evaluation protocol to test whether AIDO.Cell's latent factors are causally faithful.

---

### Section 8 — Drug Discovery: AI at Pharma Scale

**The drug discovery pipeline:**

```
  Target ID    Which gene/protein causes the disease?
      ↓        AI: GNN over PPI networks, GWAS signals
  Hit Discovery  Which molecule binds the target?
      ↓        AI: virtual screening, generative molecular design
  Lead Optimization  Improve potency, selectivity, ADMET
      ↓        AI: property prediction, multi-objective optimization
  Clinical Trials  Does it work in humans?
      ↓        AI: patient stratification, trial design
  Post-market    Long-term safety, new indications
               AI: pharmacovigilance, repurposing
```

**Recursion's approach:**
- Map millions of cellular phenotypes (images) to drugs and genetic perturbations
- Learn a "map" of biology where nearby points share mechanisms
- Drug repurposing: find drugs that move cells toward a healthy state

**Isomorphic Labs (formerly AlphaFold team):**
- IsoDDE: full structure → binding affinity → ADMET pipeline
- The promise: eliminate most of the early discovery wet lab experiments in silico
- Current limitation: structure-based, doesn't model cellular context

---

## 💡 Key Takeaways

1. **Biology is the perfect AI domain** — it has scale, hierarchy, interventional data, and clear benchmarks. Every component of the biological system is data-tractable.

2. **Foundation models have transformed biology** — ESM-2, AlphaFold, Evo 2, AIDO represent a genuine phase transition. Protein folding is "solved." Perturbation prediction is viable. Genome-scale understanding is emerging.

3. **Prediction ≠ Understanding** — every current bio FM predicts well but is causally opaque. The field is building powerful tools without knowing *why* they work or whether the representations are biologically faithful.

4. **CRL is the missing piece** — Perturbation biology provides real do-interventions. The CRL identifiability framework applies directly. The bridge paper (Sun et al., ICLR 2025) opens the door; the field hasn't fully walked through it yet.

5. **The AIDO system is the boldest bet** — connecting FMs across all biological scales mirrors the actual causal structure of biology. Whether the representations are causally identifiable is the key open question.

6. **The timeline is fast** — AlphaFold 2 was 2020. Evo 2 was 2024. The field moves in months, not years. Survey sessions like this need to be revisited quarterly.

---

## ❓ Discussion Questions

- How do you evaluate whether a bio FM truly *understands* biology vs. memorizing evolutionary patterns?
- CRISPR knockouts are real interventions — but are they the *right* interventions for CRL identifiability? What conditions must hold?
- AlphaFold "solved" protein structure prediction — what does "solved" mean in the context of perturbation prediction? What would a "solved" benchmark look like?
- Recursion has the largest proprietary cellular imaging dataset — does data scale win, or does architecture matter more?
- What is the right evaluation benchmark for causal faithfulness of biological representations?
- If you had to pick one problem space to go deepest on — protein design, perturbation prediction, or genomic FMs — which and why?

---

## 📝 DWS Preparation Checklist

### DWS-A: Foundation Dive (pre-session)
- [ ] Read AIDO master paper (Xing et al., cs.cmu.edu/~epxing/papers/2025/AIDO.pdf)
- [ ] Watch 2 FM4Bio seminar talks (Haotian Cui on scGPT + Abhinav Adduri on perturbation)
- [ ] Read Norman et al. (2019) — understand Perturb-seq experimental design
- [ ] Skim Jumper et al. (2021) AlphaFold 2 — understand the structure prediction breakthrough

### DWS-B: Frontier Exploration (pre-session)
- [ ] Read Sun et al. (ICLR 2025) — CRL from multimodal biomedical observations
- [ ] Read GEARS (Roohani 2023) — state of the art in perturbation prediction
- [ ] Skim Evo 2 paper — understand genome-scale FMs
- [ ] Survey CausCell (2025) and scCausalVI (2025)

### DWS-C: Framework Build + Demo (pre-session)
- [ ] Install requirements.txt and run Demo 01 (ESM-2 embeddings)
- [ ] Run Demo 02 (Scanpy + single-cell perturbation)
- [ ] Try AIDO.Cell quickstart on CZ Science platform
- [ ] Finalize slides and populate tex/main.tex results
