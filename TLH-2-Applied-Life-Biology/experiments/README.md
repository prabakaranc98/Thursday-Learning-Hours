# Experiments — TLH-2: Applied Life: Intelligence Meets Living Systems

> Two hands-on demos: (1) protein language model embeddings and zero-shot fitness prediction with ESM-2; (2) single-cell RNA-seq analysis and perturbation response with Scanpy + Norman 2019 dataset.

---

## Setup

```bash
pip install -r requirements.txt
```

> **GPU:** Demo 01 runs on CPU (650M model) or any GPU. Demo 02 requires ~8GB RAM.
> For full AIDO models, an A100 GPU is recommended (see AIDO.Cell quickstart on CZ Science platform).

---

## Demo 01 — ESM-2 Protein Embeddings & Zero-Shot Fitness Prediction

**File:** `demo_01_esm2_protein_embeddings.ipynb`

### Goal

Demonstrate that protein language models learn functionally meaningful representations — without any structural supervision. Use ESM-2 log-likelihoods as a zero-shot fitness predictor on real deep mutational scanning data.

### What We Do

1. **Load ESM-2** (650M parameter checkpoint from Meta AI via HuggingFace)
2. **Select a protein:** TP53 tumor suppressor (high-impact, well-studied, ~400 residues)
3. **Generate mutation set:** Sample 200 single amino acid variants from ProteinGym DMS data
4. **Compute ESM-2 embeddings** for each variant via forward pass
5. **Zero-shot fitness prediction:**
   - Compute masked marginal log-likelihood score for each variant
   - `score(mut) = log P(wt_AA | context) - log P(mut_AA | context)`
   - No fine-tuning — purely the pre-trained model's likelihood estimates
6. **Visualize:**
   - UMAP of variant embeddings — do pathogenic/benign variants separate?
   - Scatter plot: ESM-2 score vs. experimental DMS fitness
   - Compute Spearman ρ

### Expected Results

| Model | Task | Spearman ρ | Notes |
|---|---|---|---|
| ESM-2 (650M) | TP53 DMS | ~0.50–0.65 | Zero-shot, no fine-tuning |
| ESM-2 (3B) | TP53 DMS | ~0.60–0.70 | Larger model, still zero-shot |
| Supervised baseline | TP53 DMS | ~0.80+ | Fine-tuned on train split |

**Key insight:** Zero-shot ESM-2 scores already capture significant functional signal — the model learned fitness constraints from evolution alone.

### Why This Matters

This is the empirical foundation of the "evolution as pre-training" thesis. The protein LM never saw fitness measurements during training — it only saw sequences. Yet the log-likelihoods correlate with experimental outcomes because sequences that violate evolutionary constraints are also functionally deleterious.

---

## Demo 02 — Single-Cell Analysis & Perturbation Response

**File:** `demo_02_singlecell_perturbation.ipynb`

### Goal

Walk through a complete single-cell RNA-seq analysis pipeline on a Perturb-seq dataset. Understand how gene expression profiles encode cell state, how CRISPR perturbations shift these profiles, and why predicting these shifts is a hard problem that requires more than simple linear methods.

### Dataset

**Norman et al. (2019)** — combinatorial CRISPR Perturb-seq
- K562 leukemia cells
- 131 transcription factor gene knockouts (singles + pairs)
- ~111,000 cells × 11,000 genes
- We use a 15,000-cell subset for the demo

**Download:** GEO accession [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344) or via `datasets` library (loader in notebook)

### What We Do

**Part A: Preprocessing & Visualization**
1. Load AnnData object (`.h5ad` file)
2. Quality control: filter low-quality cells (min genes, max mitochondrial %)
3. Normalize: counts per 10,000 + log1p transform
4. Identify highly variable genes (HVGs) — top 2,000 by dispersion
5. PCA → UMAP: project to 2D for visualization
6. Cluster: Leiden clustering — label clusters by cell state
7. Visualize: UMAP colored by (a) cluster, (b) perturbation identity, (c) key marker genes

**Part B: Perturbation Effect Analysis**
1. Isolate control cells vs. cells with single gene knockouts
2. Compute mean expression profile per perturbation
3. PCA on perturbation mean profiles — do related knockouts cluster?
4. Differential expression: identify the top DEGs for 3 example perturbations
5. Perturbation vector: compute the "shift" in PCA space for each knockout
6. Visualize: are perturbation vectors (shifts) biologically interpretable?

**Part C: The Prediction Problem (Motivation)**
1. Hold out 10 perturbations from the dataset
2. Try the simplest baseline: predict held-out perturbation = mean control + average perturbation shift
3. Measure prediction quality: Pearson r on top 50 DEGs
4. Show why this breaks for combinatorial perturbations
5. Motivate GEARS (knowledge graph + GNN) as the right solution

### Expected Results

| Method | Task | Pearson r (top 50 DEGs) |
|---|---|---|
| Mean control (trivial baseline) | Single gene holdout | ~0.30 |
| Linear shift baseline | Single gene holdout | ~0.55–0.65 |
| scGen | Single gene holdout | ~0.70–0.80 |
| GEARS | Single gene holdout | ~0.80–0.90 |
| GEARS | Combinatorial holdout | ~0.55–0.70 |

**Key insight:** Simple linear methods work reasonably for single-gene knockouts. But combinatorial perturbations show non-linear genetic interactions — this is what makes GEARS (and causal approaches) necessary.

### Why This Matters

This demo is the entry point to the CRL × biology frontier. Every cell in the dataset experienced a specific intervention (CRISPR knockout) — that's a real do-operation. The challenge of predicting the response to unseen perturbation combinations is exactly the OOD generalization problem that CRL identifiability theorems address. A causally faithful representation would generalize to unseen combinations; a correlational one won't.

---

## Further Experiments (Optional / Advanced)

### Exp 03 — AIDO.Cell Quickstart (requires A100 GPU)
Follow the CZ Science quickstart: load AIDO.Cell, run single-cell perturbation prediction on Norman 2019, compare with scGen/GEARS baseline.
- Guide: https://virtualcellmodels.cziscience.com/quickstart/aido-quickstart

### Exp 04 — GEARS on Norman 2019 (requires GPU)
Train GEARS on Norman 2019 train split, evaluate on test set. Reproduce the main paper results on combinatorial perturbation prediction.
- Repo: https://github.com/snap-stanford/GEARS

### Exp 05 — AlphaFold Structure Prediction
Use AlphaFold Server (https://alphafoldserver.com) to predict the structure of a protein of interest. Visualize with PyMOL or ChimeraX. Compare with an ESMFold prediction.
