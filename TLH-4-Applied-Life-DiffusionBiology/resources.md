# Resources — TLH-4: Applied Life: Diffusion for Biology & Molecular Dynamics

---

## Core Libraries

| Library | Purpose |
|---|---|
| **PyTorch** | All demos |
| **torch_geometric** | Graph neural networks — required for EGNN/molecular graphs |
| **torch_scatter** | Efficient scatter operations for GNN message passing |
| **biopython** | Protein structure parsing (PDB files) |
| **mdanalysis** | Molecular dynamics trajectory analysis |
| **py3Dmol** | Interactive 3D protein visualization in Jupyter |
| **rdkit** | Cheminformatics — molecule validation, 2D drawing |
| **einops** | Tensor manipulation |

```bash
pip install torch torchvision einops tqdm matplotlib numpy scipy
pip install torch_geometric torch_scatter torch_cluster
pip install biopython mdanalysis py3Dmol rdkit-pypi
```

---

## Reference Implementations

| Repo | Description |
|---|---|
| [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) | Official Watson et al. — protein backbone design |
| [FrameDiff](https://github.com/jasonkyuyim/se3_diffusion) | Official Yim et al. — SE(3) diffusion with IPA |
| [FoldFlow](https://github.com/DreamFold/FoldFlow) | Flow matching version of FrameDiff |
| [EGNN (official)](https://github.com/vgsatorras/egnn) | E(n) Equivariant GNN — the building block for EDM |
| [EDM (official)](https://github.com/ehoogeboom/e3_diffusion_for_molecules) | E(3)-equivariant diffusion for molecule generation |
| [DiffSBDD](https://github.com/arneschneuing/DiffSBDD) | Structure-based drug design with diffusion |
| [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | Sequence design given backbone — pairs with RFdiffusion |
| [OpenFold](https://github.com/aqlaboratory/openfold) | Open-source AlphaFold 2 with IPA — reference for FrameDiff denoiser |

---

## Datasets

| Dataset | Content | Use |
|---|---|---|
| **CATH** | 500k+ protein domains with structure | RFdiffusion / FrameDiff training |
| **PDB (Protein Data Bank)** | 200k+ experimental structures | Ground truth structures |
| **QM9** | 134k small molecules with quantum properties | EDM benchmark |
| **GEOM-DRUGS** | Drug-like molecules + conformational ensembles | EDM / DiffSBDD benchmark |
| **CrossDocked2020** | Protein-ligand complexes | DiffSBDD training/benchmark |
| **MDCATH** | Molecular dynamics trajectories for CATH domains | MDGen training |

---

## Key Blog Posts and Tutorials

| Resource | Author | Why Read |
|---|---|---|
| [RFdiffusion Blog](https://www.bakerlab.org/2023/07/11/diffusion-model-for-protein-design/) | Baker Lab | Motivation + key results, accessible |
| [Protein Structure: A Visual Guide](https://proteopedia.org) | Proteopedia | Background on protein geometry |
| [EGNN Tutorial](https://medium.com/@mlforscience) | — | EGNN from scratch with code |
| [Drug Design with Diffusion](https://www.blopig.com/blog/2023/09/diffusion-models-in-drug-design/) | Oxford Protein Informatics | DiffSBDD + context for drug design |
| [MDGen Blog](https://www.microsoft.com/en-us/research/project/molecular-dynamics-ai/) | Microsoft Research | MDGen motivation + results |
| [AlphaFold Explainer](https://www.deepmind.com/research/highlighted-research/alphafold) | DeepMind | IPA module context |

---

## Tools for Protein Visualization

| Tool | Purpose |
|---|---|
| **py3Dmol** | Interactive 3D in Jupyter notebooks |
| **PyMOL** | Professional protein visualization (open-source version available) |
| **ChimeraX** | UCSF's viewer — good for large complexes |
| **Molstar** | Web-based, embeddable — used in PDB |

```python
import py3Dmol
view = py3Dmol.view(width=800, height=400)
view.addModel(open('protein.pdb').read(), 'pdb')
view.setStyle({'cartoon': {'color': 'spectrum'}})
view.show()
```

---

## Pre-trained Models

| Model | Access | Notes |
|---|---|---|
| **RFdiffusion** | GitHub + HuggingFace | Full backbone design model |
| **ProteinMPNN** | GitHub | Sequence design given backbone |
| **ESM-2 (650M)** | `facebook/esm2_t33_650M_UR50D` | Protein language model — embeddings |
| **ESM-IF1** | `facebook/esm_if1_gvp4_t16_142M_UR50` | Inverse folding — sequence from structure |
| **OpenFold** | GitHub | AlphaFold 2 weights + training code |
