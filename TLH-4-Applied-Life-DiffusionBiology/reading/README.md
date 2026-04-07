# Reading List — TLH-4: Applied Life: Diffusion for Biology & Molecular Dynamics

> 30 papers across 5 themes. ⭐ = essential pre-session reading (read these 5 minimum).
> Estimated total: ~26 hrs. Pre-session essentials: ~4.5 hrs.

---

## Theme 1: Protein Backbone Diffusion

*Diffusion over SE(3) — the foundational protein design papers.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 1 | **De novo design of protein structure and function with RFdiffusion** | Watson, Juergens, Bennett et al. | Nature 2023 | The paper. RoseTTAFold as denoiser + IGSO(3) diffusion. Wet lab validation. ~2 hrs |
| ⭐ 2 | **SE(3) Diffusion Model with Application to Protein Backbone Generation (FrameDiff)** | Yim, Campbell et al. | ICML 2023 | Cleaner derivation of SE(3) diffusion. IPA denoiser. Benchmarked with RFdiffusion. |
| 3 | **Protein Design with Guided Discrete Diffusion** | — | 2024 | Joint sequence + structure diffusion |
| 4 | **Chroma: A Generative Model of Protein Structure** | Ingraham et al. | NeurIPS 2023 | Diffusion over full-atom protein representations |
| 5 | **FoldFlow: De Novo Protein Structure Generation with Flow Matching** | Bose et al. | 2023 | FM version of FrameDiff — preview of TLH-2 × TLH-4 intersection |

---

## Theme 2: Small Molecule Generation

*Equivariant diffusion for 3D drug-like molecules.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 6 | **Equivariant Diffusion for Molecule Generation in 3D (EDM)** | Hoogeboom, Satorras, Vignac, Welling | ICML 2022 | E(3)-equivariant DDPM for 3D molecular graph generation. The key molecule paper. |
| 7 | **GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation** | Xu et al. | ICLR 2022 | Diffusion over molecular conformations — geometry without atom type generation |
| 8 | **GDSS: Score-based Generative Modeling of Graphs** | Jo et al. | ICML 2022 | Score matching on graphs — nodes + edges jointly |
| 9 | **MiDi: Joint Generation of 3D Molecules and Conformers** | Vignac et al. | 2023 | Joint 2D graph + 3D conformation generation |
| 10 | **DiffSBDD: Structure-Based Drug Design with Equivariant Diffusion** | Schneuing et al. | NeurIPS 2023 | Pocket-conditioned ligand generation — drug design application |

---

## Theme 3: Equivariant Neural Networks

*The architectural foundation — understanding E(3)-equivariance.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 11 | **E(n) Equivariant Graph Neural Networks (EGNN)** | Satorras, Hoogeboom, Welling | ICML 2021 | The architecture used in EDM. Simple and effective E(3)-equivariant GNN. |
| 12 | **SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks** | Fuchs et al. | NeurIPS 2020 | Equivariant attention via spherical harmonics |
| 13 | **Equivariant Message Passing for the Prediction of Tensorial Properties** | Batzner et al. | NeurIPS 2022 | NequIP — equivariant GNN for molecular energy prediction |
| 14 | **Highly Accurate Protein Structure Prediction with AlphaFold** | Jumper et al. | Nature 2021 | AlphaFold 2 — invariant point attention (IPA) — the denoiser architecture in FrameDiff |
| 15 | **Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges** | Bronstein et al. | 2021 | The theoretical foundation — read Ch. 3-5 for groups and equivariance |

---

## Theme 4: Molecular Dynamics and Conformational Sampling

*Diffusion for simulating protein dynamics.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| 16 | **MDGen: Simulation-Free Protein Molecular Dynamics with Generative Model** | — | 2024 | Diffusion over MD trajectories — generate conformational ensembles without MD |
| 17 | **Timewarp: Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics** | Klein et al. | NeurIPS 2023 | Learn coarse-grained MD dynamics — complementary approach |
| 18 | **Boltz-1: Democratizing Biomolecular Interaction Modeling** | Wohlwend et al. | 2024 | AlphaFold 3-style structure prediction + FM |
| 19 | **DiffusionProtein: Diffusion for Protein Conformation Sampling** | — | 2024 | Diffusion specifically for sampling near-equilibrium conformations |
| 20 | **DiffBP: Generative Diffusion of 3D Molecules for Target Protein Binding** | Lin & Zhang | 2022 | Binding pose generation for drug design |

---

## Theme 5: Broader Biological Diffusion Applications

*Antibodies, RNA, DNA, and beyond images.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| 21 | **RFdiffusion All-Atom: General Protein Design with RFdiffusion** | Krishna et al. | Science 2024 | Extension of RFdiffusion to all-atom including ligands and metals |
| 22 | **AbDiffuser: Full-Atom Generation of in vitro Antibodies** | Martinkus et al. | 2023 | Antibody CDR loop generation with diffusion |
| 23 | **RNA Design with Diffusion Models** | — | 2024 | Applying backbone diffusion to RNA structural design |
| 24 | **Genie2: Generative Modeling of Functional Protein Design** | — | 2024 | Large-scale protein design conditioned on functional annotations |
| 25 | **AlphaFold 3** | Abramson et al. | Nature 2024 | AF3 uses diffusion module for all-atom structure prediction — landmark paper |
| 26 | **EvoBind: In Silico Directed Evolution of Peptide Binders** | — | 2024 | Diffusion + RL for peptide binder optimization |
| 27 | **Navigating the Design Space of Equivariant Diffusion Models** | — | 2024 | Systematic study of SE(3)-equivariant diffusion architectures |
| 28 | **Score-Based Generative Modeling on Protein Sequences** | — | 2023 | Discrete diffusion / score matching for amino acid sequences |
| 29 | **Causal Representation Learning for Biological Perturbation** | Sun, Zhang, Xing | ICLR 2025 | CRL identifiability with CRISPR as interventions — bridge to TLH-6 |
| 30 | **Flow Matching on General Geometries** | Chen & Lipman | ICLR 2024 | Riemannian FM — the unified theory for protein/molecule manifolds |

---

## Suggested Reading Order

**Before the session (~4.5 hrs, ⭐ only):**
1. **EGNN** (#11) — 45 min — understand equivariant GNNs before reading diffusion papers
2. **EDM** (#6) — 90 min — E(3)-equivariant DDPM for molecules. The clearest equivariant diffusion paper.
3. **RFdiffusion** (#1) — 90 min — protein backbone diffusion. Read the Methods section carefully.
4. **FrameDiff** (#2) — 45 min — cleaner theory of SE(3) diffusion
5. **AlphaFold 3** (#25) — 30 min — skim architecture section — diffusion in the context of the best structure predictor

**To go deeper:**
- Theory: Geometric Deep Learning (#15) Ch. 3-5 → EGNN (#11) → SE(3)-Transformers (#12)
- Molecules: EDM (#6) → GeoDiff (#7) → DiffSBDD (#10) → MiDi (#9)
- Dynamics: MDGen (#16) → Timewarp (#17) for the MD acceleration arc

**TLH-6 bridge:**
- Sun, Zhang, Xing ICLR 2025 (#29) — CRL × biological perturbations connects protein editing to causal representation learning
