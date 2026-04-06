# Links — TLH-2: Applied Life: Intelligence Meets Living Systems

> 80+ curated reference links organized by theme.

---

## Biological Foundation Models

### ESM / EvolutionaryScale (Meta AI)
- [ESM-2 Paper (Lin et al., 2023)](https://www.science.org/doi/10.1126/science.ade2574) — Evolutionary-scale prediction of atomic-level protein structure
- [ESM GitHub](https://github.com/facebookresearch/esm) — official code repository
- [ESM HuggingFace](https://huggingface.co/facebook/esm2_t33_650M_UR50D) — pre-trained checkpoints
- [ESM-3 Paper (Hayes et al., 2024)](https://www.evolutionaryscale.ai/blog/esm3-release) — multimodal protein model
- [ESMFold: Fast structure prediction](https://esmatlas.com/resources?action=fold)

### GenBio AI / AIDO
- [AIDO Master Paper](https://cs.cmu.edu/~epxing/papers/2025/AIDO.pdf) — the full system description
- [GenBio AI website](https://genbio.ai) — overview of all models
- [AIDO GitHub](https://github.com/genbio-ai/AIDO) — full codebase
- [AIDO on HuggingFace](https://huggingface.co/genbio-ai) — all pre-trained checkpoints
- [AIDO.ModelGenerator Docs](https://genbio-ai.github.io/ModelGenerator) — plug-and-play framework
- [FM4Bio Seminar Series](https://www.youtube.com/watch?v=4d9BcFIOEEI&list=PL6nJuBRU_qT0QjtD3XCVuHALK26p-HZCZ) — GenBio AI YouTube lecture series
- [AIDO.Cell Quickstart (CZ Science)](https://virtualcellmodels.cziscience.com/quickstart/aido-quickstart)

### Evo 2 / Arc Institute
- [Evo 2 Paper (Nature, 2024)](https://www.nature.com/articles/s41586-026-10176-5) — genome-scale foundation model
- [Arc Institute](https://arcinstitute.org) — research institute behind Evo 2
- [Evo 2 GitHub](https://github.com/ArcInstitute/evo2) — model code and weights

### Geneformer / scGPT
- [Geneformer Paper (Theodoris et al., 2023)](https://www.nature.com/articles/s41586-023-06139-9) — transfer learning in network biology
- [Geneformer HuggingFace](https://huggingface.co/ctheodoris/Geneformer)
- [scGPT Paper (Cui et al., 2024)](https://www.nature.com/articles/s41592-024-02201-0)
- [scGPT GitHub](https://github.com/bowang-lab/scGPT)

---

## Protein Structure & Design

### AlphaFold
- [AlphaFold 2 Paper (Jumper et al., 2021)](https://www.nature.com/articles/s41586-021-03819-2) — Nobel Prize 2024
- [AlphaFold 3 Paper (Evans et al., 2024)](https://www.nature.com/articles/s41586-024-07487-w)
- [AlphaFold Database](https://alphafold.ebi.ac.uk) — 200M structures
- [AlphaFold Server](https://alphafoldserver.com) — predict structures online
- [OpenFold GitHub](https://github.com/aqlaboratory/openfold) — open PyTorch reimplementation
- [Isomorphic Labs](https://isomorphiclabs.com) — drug design using AlphaFold lineage

### Baker Lab — Protein Design
- [RFdiffusion Paper (Watson et al., 2023)](https://www.nature.com/articles/s41586-023-06415-8)
- [RFdiffusion GitHub](https://github.com/RosettaCommons/RFdiffusion)
- [ProteinMPNN Paper (Dauparas et al., 2022)](https://www.science.org/doi/10.1126/science.add2187)
- [ProteinMPNN GitHub](https://github.com/dauparas/ProteinMPNN)
- [RoseTTAFold Paper (Baek et al., 2021)](https://www.science.org/doi/10.1126/science.abj8754)
- [RoseTTAFold GitHub](https://github.com/RosettaCommons/RoseTTAFold)

### Protein Benchmarks
- [ProteinGym Website](https://proteingym.org) — 250 deep mutational scanning datasets
- [ProteinGym GitHub](https://github.com/OATML-Markslab/ProteinGym)
- [FLIP Benchmark](https://github.com/J-SNACKKB/FLIP) — fitness landscape inference
- [Chroma: Generative protein design](https://github.com/generatebio/chroma)

---

## Single-Cell Biology & Analysis

### Core Tools
- [Scanpy Documentation](https://scanpy.readthedocs.io)
- [Scanpy GitHub](https://github.com/scverse/scanpy)
- [Scanpy Tutorials](https://scanpy.readthedocs.io/en/stable/tutorials.html)
- [AnnData Documentation](https://anndata.readthedocs.io)
- [scVI-tools Documentation](https://docs.scvi-tools.org)
- [scVI Paper (Lopez et al., 2018)](https://www.nature.com/articles/s41592-018-0229-2)

### Atlases & Infrastructure
- [CellxGene Browser](https://cellxgene.cziscience.com)
- [CellxGene Census GitHub](https://github.com/chanzuckerberg/cellxgene-census)
- [CZ Science Virtual Cells Platform](https://virtualcellmodels.cziscience.com)
- [Human Cell Atlas](https://www.humancellatlas.org)

---

## Perturbation Biology

### Foundational Perturb-seq Papers
- [Norman et al. (2019) — Combinatorial Perturb-seq](https://www.science.org/doi/10.1126/science.aax4438) — genetic interaction manifolds
- [Replogle et al. (2022) — Genome-scale Perturb-seq](https://www.cell.com/cell/fulltext/S0092-8674(22)00597-9) — 9,000+ knockouts
- [Dixit et al. (2016) — CRISP-seq](https://www.cell.com/cell/fulltext/S0092-8674(16)31610-5) — original Perturb-seq paper

### Perturbation Prediction Models
- [GEARS Paper (Roohani et al., 2023)](https://www.nature.com/articles/s41587-023-01905-6)
- [GEARS GitHub](https://github.com/snap-stanford/GEARS)
- [scGen Paper (Lotfollahi et al., 2019)](https://www.nature.com/articles/s41592-019-0494-8)
- [scGen GitHub](https://github.com/theislab/scgen)
- [CellOT Paper (Bunne et al., 2023)](https://www.nature.com/articles/s41592-023-01969-x)
- [CellOT GitHub](https://github.com/bunnech/cellot)
- [CINEMA-OT GitHub](https://github.com/vandijklab/CINEMA-OT)
- [Pertpy GitHub](https://github.com/theislab/pertpy)

---

## Causal Representation Learning for Biology

### CRL Theory
- [Schölkopf et al. (2021) — Toward Causal Representation Learning](https://arxiv.org/abs/2102.11107)
- [Locatello et al. (2019) — Challenging Common Assumptions in CRL](https://arxiv.org/abs/1811.12359)
- [Ahuja et al. (2022) — Weakly Supervised CRL with Intervention Pairs](https://arxiv.org/abs/2208.00658)
- [Brehmer et al. (2022) — Weakly Supervised Causal Representation Learning](https://arxiv.org/abs/2203.16437)

### CRL × Biology Bridge
- [Sun, Zhang, Xing et al. (ICLR 2025) — CRL from Multimodal Biomedical Observations](https://openreview.net/forum?id=XgWhNUNWXJ) — THE bridge paper
- [CausalBench Paper (Chevalley et al., 2023)](https://arxiv.org/abs/2210.17283)
- [CausalBench GitHub](https://github.com/causalbench/causalbench)
- [DREAM5 Network Inference Benchmark](https://dreamchallenges.org/dream5)
- [sVAE+ / Sparse Mechanism Shift for single-cell (Lopez et al.)](https://arxiv.org/abs/2301.01004)

---

## Drug Discovery AI

### Key Papers
- [Stokes et al. (2020) — Deep learning for antibiotic discovery](https://www.cell.com/cell/fulltext/S0092-8674(20)30102-1)
- [Huang et al. (2022) — Therapeutics Data Commons](https://www.cell.com/patterns/fulltext/S2666-3899(22)00232-0)
- [Gómez-Bombarelli et al. (2018) — Chemical design with continuous molecule representation](https://pubs.acs.org/doi/10.1021/acscentsci.7b00572)
- [Angermueller et al. (2019) — RL for biological sequence design](https://arxiv.org/abs/1901.11695)

### Tools & Databases
- [Therapeutics Data Commons (TDC)](https://tdcommons.ai)
- [DeepChem GitHub](https://github.com/deepchem/deepchem)
- [REINVENT 4 GitHub](https://github.com/MolecularAI/REINVENT4)
- [ChEMBL Database](https://www.ebi.ac.uk/chembl)
- [BindingDB](https://www.bindingdb.org)
- [PubChem](https://pubchem.ncbi.nlm.nih.gov)
- [Recursion Pharmaceuticals](https://www.recursion.com)

---

## Synthetic Biology & Computational Neuroscience

### Synthetic Biology AI
- [Inukai et al. (2024) — LLMs for synthetic biology](https://www.nature.com/articles/s41587-024-02442-y)
- [iGEM Foundation](https://igem.org) — synthetic biology community
- [Addgene — plasmid repository](https://www.addgene.org)

### Computational Neuroscience
- [Allen Brain Atlas](https://atlas.brain-map.org) — comprehensive brain atlases
- [DANDI Archive](https://dandiarchive.org) — neurophysiology data repository
- [NeuralData.org](https://neuraldata.org) — neural data tools and datasets
- [BrainGlobe](https://brainglobe.info) — Python tools for computational neuroanatomy
- [Connectome Workbench (HCP)](https://www.humanconnectome.org/software/connectome-workbench)

---

## Key Labs & Companies

| Entity | Focus | Link |
|---|---|---|
| GenBio AI | Multi-scale bio FMs | [genbio.ai](https://genbio.ai) |
| DeepMind / Isomorphic Labs | Protein + drug design | [deepmind.google/science/alphafold](https://deepmind.google/science/alphafold) |
| Arc Institute | Genome-scale FMs | [arcinstitute.org](https://arcinstitute.org) |
| Recursion | AI drug discovery | [recursion.com](https://www.recursion.com) |
| Bioptimus | Universal bio FM | [bioptimus.com](https://bioptimus.com) |
| CZ Science | Virtual cells, open bio | [chanzuckerberg.com/science](https://chanzuckerberg.com/science) |
| Baker Lab (UW) | Protein design | [bakerlab.org](https://www.bakerlab.org) |
| Theislab (Helmholtz) | Single-cell analysis | [theislab.github.io](https://theislab.github.io) |
| CMU-CLeaR (Kun Zhang) | CRL theory | [kunzhang.org](https://www.andrew.cmu.edu/user/kunz1) |
| Broad Institute — CLeaR | CRL for biology workshop | [broadinstitute.org](https://www.broadinstitute.org) |

---

## Venues & Workshops

- [CLeaR Workshop (Causal Learning & Reasoning)](https://www.cclear.cc) — annual at ICLR
- [NeurIPS CRL Workshop](https://crl-workshop.github.io)
- [ICLR LMRL Workshop (Learning Meaningful Representations of Life)](https://www.lmrl.org)
- [Nature Methods](https://www.nature.com/nmeth) — primary venue for computational biology tools
- [Nature Biotechnology](https://www.nature.com/nbt) — applied bio methods
- [Cell Systems](https://www.cell.com/cell-systems/home) — computational systems biology
