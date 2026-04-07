# Reference Links — TLH-4: Applied Life: Diffusion for Biology & Molecular Dynamics

---

## Protein Backbone Diffusion

- https://www.nature.com/articles/s41586-023-06415-8 — Watson et al. (2023) — RFdiffusion (Nature)
- https://arxiv.org/abs/2302.02277 — Yim et al. (2023) — FrameDiff: SE(3) Diffusion for Protein Backbone
- https://arxiv.org/abs/2306.12509 — Bose et al. (2023) — FoldFlow
- https://arxiv.org/abs/2312.04557 — Bose et al. (2024) — SE(3)-SFM
- https://arxiv.org/abs/2212.04048 — Ingraham et al. (2023) — Chroma
- https://arxiv.org/abs/2312.02598 — Genie2 (2024)
- https://www.science.org/doi/10.1126/science.adi1910 — Krishna et al. (2024) — RFdiffusion All-Atom

## Small Molecule Generation

- https://arxiv.org/abs/2203.17003 — Hoogeboom et al. (2022) — EDM: E(3)-Equivariant Diffusion for Molecules
- https://arxiv.org/abs/2203.02923 — Xu et al. (2022) — GeoDiff
- https://arxiv.org/abs/2209.14734 — Schneuing et al. (2022) — DiffSBDD
- https://arxiv.org/abs/2208.09530 — Jo et al. (2022) — GDSS: Score-based Molecular Graph Generation
- https://arxiv.org/abs/2301.10882 — Vignac et al. (2023) — MiDi: 3D Molecule + Conformation

## Equivariant Neural Networks

- https://arxiv.org/abs/2102.09844 — Satorras, Hoogeboom, Welling (2021) — EGNN
- https://arxiv.org/abs/2006.10503 — Fuchs et al. (2020) — SE(3)-Transformers
- https://arxiv.org/abs/2101.03164 — Batzner et al. (2022) — NequIP / e3nn
- https://arxiv.org/abs/2104.09864 — Jumper et al. (2021) — AlphaFold 2 (IPA module)
- https://arxiv.org/abs/2309.07473 — Abramson et al. (2024) — AlphaFold 3
- https://arxiv.org/abs/2104.02848 — Bronstein et al. (2021) — Geometric Deep Learning

## Molecular Dynamics and Conformational Sampling

- https://arxiv.org/abs/2402.04619 — MDGen (2024) — Simulation-Free MD with Diffusion
- https://arxiv.org/abs/2302.01170 — Klein et al. (2023) — Timewarp
- https://arxiv.org/abs/2310.01500 — Wohlwend et al. (2024) — Boltz-1
- https://www.mdanalysis.org — MDAnalysis library documentation
- https://www.mdtraj.org — MDTraj trajectory analysis

## Drug Design and Applications

- https://arxiv.org/abs/2209.14734 — Schneuing et al. (2022) — DiffSBDD
- https://arxiv.org/abs/2210.05433 — Lin & Zhang (2022) — DiffBP
- https://arxiv.org/abs/2302.07271 — Martinkus et al. (2023) — AbDiffuser (antibodies)
- https://arxiv.org/abs/2309.12929 — Genie2 for protein design

## CRL × Biology Bridge (TLH-6 Preview)

- https://arxiv.org/abs/2502.09290 — Sun, Zhang, Xing (ICLR 2025) — CRL Identifiability for Biological Perturbations
- https://arxiv.org/abs/2205.12628 — Norman et al. (2019) — Perturb-seq dataset (CRISPR knockouts)

## Official Code Repositories

- https://github.com/RosettaCommons/RFdiffusion — RFdiffusion official
- https://github.com/jasonkyuyim/se3_diffusion — FrameDiff official
- https://github.com/DreamFold/FoldFlow — FoldFlow official
- https://github.com/vgsatorras/egnn — EGNN official
- https://github.com/ehoogeboom/e3_diffusion_for_molecules — EDM official
- https://github.com/arneschneuing/DiffSBDD — DiffSBDD official
- https://github.com/dauparas/ProteinMPNN — ProteinMPNN (pairs with RFdiffusion)
- https://github.com/aqlaboratory/openfold — OpenFold (AlphaFold 2 open-source)

## HuggingFace Resources

- https://huggingface.co/facebook/esm2_t33_650M_UR50D — ESM-2 protein LM
- https://huggingface.co/facebook/esm_if1_gvp4_t16_142M_UR50 — ESM-IF1 inverse folding
- https://huggingface.co/facebook/esmfold_v1 — ESMFold structure predictor

## Tutorials and Blog Posts

- https://www.bakerlab.org/2023/07/11/diffusion-model-for-protein-design/ — RFdiffusion blog
- https://www.blopig.com/blog/2023/09/diffusion-models-in-drug-design/ — Drug design with diffusion
- https://proteopedia.org/wiki/index.php/Introduction_to_protein_structure — Protein structure primer
- https://www.ebi.ac.uk/training/online/courses/protein-structure-and-function — EBI protein structure course

## Benchmarks

- https://www.rcsb.org — Protein Data Bank (PDB) — ground truth structures
- https://paperswithcode.com/task/molecule-generation — Molecule generation benchmarks
- https://paperswithcode.com/task/protein-design — Protein design benchmarks
- https://www.cameo3d.org — Continuous Automated Model EvaluatiOn for structure prediction
