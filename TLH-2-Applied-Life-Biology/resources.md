# Resources & Tools — TLH-2: Applied Life: Intelligence Meets Living Systems

> Curated ecosystem: foundation models, analysis frameworks, datasets, benchmarks, and community resources for AI + Biology.

---

## Biological Foundation Models

| Tool / Model | Description | Links |
|---|---|---|
| **ESM-2 / ESM-3** (Meta AI) | Protein language models — 8M to 15B params, trained on 250M sequences | [GitHub](https://github.com/facebookresearch/esm) · [HuggingFace](https://huggingface.co/facebook/esm2_t33_650M_UR50D) |
| **ESMFold** (Meta AI) | Fast structure prediction from ESM-2 — no MSA required, 60× faster than AF2 | [GitHub](https://github.com/facebookresearch/esm) |
| **AIDO** (GenBio AI) | Multi-scale FM system: DNA, RNA, Protein, Cell, Tissue, StructureDiffusion | [GitHub](https://github.com/genbio-ai/AIDO) · [HuggingFace](https://huggingface.co/genbio-ai) |
| **AIDO.ModelGenerator** | Plug-and-play framework for adapting AIDO models to downstream tasks | [Docs](https://genbio-ai.github.io/ModelGenerator) |
| **Geneformer** | Single-cell FM using rank-value gene expression encoding | [HuggingFace](https://huggingface.co/ctheodoris/Geneformer) |
| **scGPT** | Generative pre-training for single-cell multi-omics | [GitHub](https://github.com/bowang-lab/scGPT) |
| **Evo 2** (Arc Institute) | Genome-scale DNA FM — 9T bp, 1M context, all domains of life | [GitHub](https://github.com/ArcInstitute/evo2) |
| **Nucleotide Transformer** (Bioptimus) | DNA FM with 12 pre-trained models for human and multi-species genomics | [GitHub](https://github.com/instadeepai/nucleotide-transformer) |
| **CaLM** (Caltech) | Codon-level language model for mRNA and expression optimization | [GitHub](https://github.com/oxpig/CaLM) |

---

## Protein Structure & Design

| Tool | Description | Links |
|---|---|---|
| **AlphaFold 2/3** (DeepMind) | Structure prediction — 200M structures freely available | [AlphaFold Server](https://alphafoldserver.com) · [DB](https://alphafold.ebi.ac.uk) |
| **OpenFold** | Open-source reimplementation of AlphaFold 2 in PyTorch | [GitHub](https://github.com/aqlaboratory/openfold) |
| **RFdiffusion** (Baker Lab) | Diffusion model over protein backbone coordinates for de novo design | [GitHub](https://github.com/RosettaCommons/RFdiffusion) |
| **ProteinMPNN** (Baker Lab) | Inverse folding — design sequence given 3D backbone | [GitHub](https://github.com/dauparas/ProteinMPNN) |
| **RoseTTAFold** (Baker Lab) | Three-track structure prediction network | [GitHub](https://github.com/RosettaCommons/RoseTTAFold) |
| **Chroma** (Generate Biomedicines) | Generative protein design via DDPM | [GitHub](https://github.com/generatebio/chroma) |
| **ProteinGym** | Benchmark for protein fitness prediction — 250 deep mutational scanning datasets | [GitHub](https://github.com/OATML-Markslab/ProteinGym) · [Website](https://proteingym.org) |
| **PyMOL** | Protein structure visualization — molecular graphics | [Website](https://pymol.org) |
| **ChimeraX** (UCSF) | Molecular visualization and analysis | [Website](https://www.rbvi.ucsf.edu/chimerax) |
| **MDAnalysis** | Python library for molecular dynamics trajectory analysis | [GitHub](https://github.com/MDAnalysis/mdanalysis) |

---

## Single-Cell Biology & Analysis

| Tool | Description | Links |
|---|---|---|
| **Scanpy** | Single-cell analysis in Python — preprocessing, PCA, UMAP, clustering | [GitHub](https://github.com/scverse/scanpy) · [Docs](https://scanpy.readthedocs.io) |
| **AnnData** | Annotated data matrix — the standard container for single-cell data | [GitHub](https://github.com/scverse/anndata) |
| **scVI-tools** | Probabilistic models for single-cell data — scVI, scANVI, totalVI, and more | [GitHub](https://github.com/scverse/scvi-tools) · [Docs](https://docs.scvi-tools.org) |
| **CellxGene** (CZ Science) | Single-cell visualization and atlas browser | [Website](https://cellxgene.cziscience.com) |
| **CellxGene Census** | 56M+ cells from 700+ datasets — one unified API | [GitHub](https://github.com/chanzuckerberg/cellxgene-census) |
| **CZ Virtual Cells Platform** | Hosts AIDO.Cell quickstart and virtual cell infrastructure | [Website](https://virtualcellmodels.cziscience.com) |
| **Seurat** (R) | Comprehensive scRNA-seq analysis toolkit | [Website](https://satijalab.org/seurat) |
| **Harmony** | Fast, scalable batch correction for single-cell data | [GitHub](https://github.com/immunogenomics/harmony) |

---

## Perturbation Biology Tools

| Tool | Description | Links |
|---|---|---|
| **GEARS** | Knowledge graph + GNN for combinatorial perturbation prediction | [GitHub](https://github.com/snap-stanford/GEARS) |
| **scGen** | VAE-based perturbation response prediction | [GitHub](https://github.com/theislab/scgen) |
| **CellOT** | Optimal transport for single-cell perturbation modeling | [GitHub](https://github.com/bunnech/cellot) |
| **CINEMA-OT** | Causal-inspired: separates shared vs. treatment-specific perturbation effects | [GitHub](https://github.com/vandijklab/CINEMA-OT) |
| **Pertpy** | Unified perturbation analysis toolkit | [GitHub](https://github.com/theislab/pertpy) |
| **sVAE+ / scCausalVI** | Causal VAE approaches for single-cell perturbation | — |

---

## Drug Discovery & Molecular Design

| Tool | Description | Links |
|---|---|---|
| **DeepChem** | Deep learning for drug discovery — molecules, proteins, materials | [GitHub](https://github.com/deepchem/deepchem) · [Docs](https://deepchem.readthedocs.io) |
| **RDKit** | Cheminformatics and molecular visualization — industry standard | [GitHub](https://github.com/rdkit/rdkit) · [Docs](https://www.rdkit.org) |
| **REINVENT 4** | Generative chemistry for drug design — SMILES-based RL | [GitHub](https://github.com/MolecularAI/REINVENT4) |
| **Therapeutics Data Commons (TDC)** | Machine learning datasets and tasks for therapeutics | [Website](https://tdcommons.ai) · [GitHub](https://github.com/mims-harvard/TDC) |
| **MolGAN** | Implicit generative model for small molecular graphs | [GitHub](https://github.com/nicola-decao/MolGAN) |
| **ChEMBL** | Bioactivity database — compounds, targets, assays | [Website](https://www.ebi.ac.uk/chembl) |
| **BindingDB** | Database of protein-ligand binding affinities | [Website](https://www.bindingdb.org) |

---

## Key Datasets

| Dataset | Description | Access |
|---|---|---|
| **UniProt / Swiss-Prot** | 250M+ protein sequences, 570K manually reviewed | [uniprot.org](https://www.uniprot.org) |
| **Protein Data Bank (PDB)** | 200,000+ experimentally determined protein structures | [rcsb.org](https://www.rcsb.org) |
| **CellxGene Census** | 56M single cells, 700+ datasets, unified API | [cellxgene.cziscience.com](https://cellxgene.cziscience.com) |
| **Norman et al. 2019** | Combinatorial Perturb-seq: 131×131 gene KO combinations | [GEO: GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344) |
| **Replogle et al. 2022** | Genome-scale Perturb-seq: 9,000+ knockouts, 2.5M cells | [GEO: GSE188836](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE188836) |
| **LINCS L1000** | Drug perturbation signatures across 978 landmark genes | [lincsportal.ccs.miami.edu](https://lincsportal.ccs.miami.edu) |
| **DepMap** | Cancer dependency map — gene essentiality across cancer cell lines | [depmap.org](https://depmap.org) |
| **ProteinGym** | 250 deep mutational scanning datasets for fitness benchmarking | [proteingym.org](https://proteingym.org) |
| **AlphaFold DB** | 200M+ predicted protein structures from AlphaFold 2 | [alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk) |

---

## Benchmarks

| Benchmark | Domain | What It Tests |
|---|---|---|
| **ProteinGym** | Protein fitness | Zero-shot fitness prediction on 250 DMS datasets |
| **FLIP** | Protein fitness | Fitness landscape inference — 5 supervised tasks |
| **PEER** | Molecular properties | 15 molecular property prediction tasks |
| **CausalBench** | Gene regulatory networks | GRN inference from Perturb-seq; causal structure recovery |
| **DREAM5** | Network inference | Gene regulatory network inference — community benchmark |
| **TDC Leaderboard** | Drug discovery | ADMET, binding, generation, reaction prediction |
| **sci-Plex / Norman 2019** | Perturbation prediction | Correlation between predicted and measured responses |
| **CASP** | Protein structure | Bi-annual blind structure prediction contest |

---

## Community & Courses

| Resource | Type | Link |
|---|---|---|
| **FM4Bio Seminar Series** (GenBio AI) | Video lecture series | [YouTube Playlist](https://www.youtube.com/watch?v=4d9BcFIOEEI&list=PL6nJuBRU_qT0QjtD3XCVuHALK26p-HZCZ) |
| **CLeaR Workshop** (Broad Institute) | Annual workshop on CRL for biology | Yearly at NeurIPS / ICLR |
| **Scanpy Tutorials** | Hands-on single-cell analysis | [scanpy.readthedocs.io/tutorials](https://scanpy.readthedocs.io/en/stable/tutorials.html) |
| **scVI-tools Tutorials** | Probabilistic single-cell modeling | [docs.scvi-tools.org](https://docs.scvi-tools.org/en/stable/tutorials/index.html) |
| **DeepChem Tutorials** | Drug discovery ML | [deepchem.readthedocs.io](https://deepchem.readthedocs.io/en/latest/tutorials/) |
| **Computational Biology (MIT 7.91J)** | Course | MIT OpenCourseWare |

---

## Experiment Stack (requirements.txt)

```
Python 3.11+
torch>=2.0.0
transformers>=4.41.0
fair-esm>=2.0.0          # ESM-2 from Meta
scanpy>=1.9.0
anndata>=0.9.0
scvi-tools>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
umap-learn>=0.5.3
scikit-learn>=1.3.0
scipy>=1.11.0
biopython>=1.81
rdkit>=2023.9.1
deepchem>=2.7.1
jupyter>=1.0.0
tqdm>=4.65.0
einops>=0.7.0
accelerate>=0.24.0
datasets>=2.14.0
plotly>=5.17.0
```
