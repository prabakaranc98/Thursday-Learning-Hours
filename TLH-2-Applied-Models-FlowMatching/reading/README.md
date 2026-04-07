# Reading List — TLH-2: Applied Models: Flow Matching

> 30 papers across 5 themes. ⭐ = essential pre-session reading (read these 5 minimum).
> Estimated total: ~22 hrs. Pre-session essentials: ~4 hrs.

---

## Theme 1: Flow Matching Foundations

*The core papers — linear paths, conditional flow matching, stochastic interpolants.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 1 | **Flow Matching for Generative Modeling** | Lipman, Chen, Ben-Hamu, Nickel, Le | ICLR 2023 | The CFM paper — marginal vs conditional flow matching, optimal transport paths. Start here. ~1.5 hrs |
| ⭐ 2 | **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow** | Liu, Gong, Liu | ICLR 2023 | Rectified flow — linear interpolation, reflow for straightening, 1-step distillation. ~1 hr |
| 3 | **Building Normalizing Flows with Stochastic Interpolants** | Albergo & Vanden-Eijnden | ICLR 2023 | Unified theory — generalizes both diffusion and flow matching. ~1 hr |
| 4 | **Improving and Generalizing Flow-Matching Models** | Albergo, Boffi, Vanden-Eijnden | 2023 | Extensions: stochastic interpolants with score conditioning |
| 5 | **Action Matching: Learning Stochastic Dynamics from Samples** | Neklyudov et al. | ICML 2023 | Flow matching for non-equilibrium dynamics — thermodynamics connection |

---

## Theme 2: Optimal Transport in Flow Matching

*Why OT paths are better than naive linear interpolation.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 6 | **Improving and Generalizing Flow-Matching Models** | Albergo et al. | 2023 | OT-FM — use OT map $T$ to pair $(x_0, x_1)$ optimally → straighter paths |
| 7 | **Multisample Flow Matching: Straightening Flows with Minibatch Couplings** | Pooladian et al. | ICML 2023 | Minibatch OT approximation for practical CFM training |
| 8 | **Flow Matching on General Geometries** | Chen & Lipman | ICLR 2024 | CFM on Riemannian manifolds — generalizes to protein geometry |
| 9 | **Discrete Flow Matching** | Campbell et al. | NeurIPS 2024 | FM for discrete data (tokens, sequences) — the bridge to language models |
| 10 | **Categorical Flow Matching on Statistical Manifolds** | Yingjie Shi et al. | NeurIPS 2024 | Flow matching for categorical distributions |

---

## Theme 3: Production Models — SD3 and FLUX

*How flow matching powers frontier text-to-image models.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 11 | **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3)** | Esser et al. (Stability AI) | CVPR 2024 | MM-DiT + rectified flow + T5-XXL text encoder → Stable Diffusion 3 |
| 12 | **FLUX.1** | Black Forest Labs | 2024 | Open-weight flow matching model — successor to SD |
| 13 | **SiT: Scalable Interpolant Transformers** | Ma et al. | 2024 | DiT backbone + stochastic interpolants → state-of-the-art FID on ImageNet |
| 14 | **Simple Diffusion: End-to-End Diffusion for High Resolution Images** | Hoogeboom et al. | ICML 2023 | Multiscale noise schedule + simple architecture — baseline comparison |
| 15 | **Würstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models** | Pernias et al. | ICLR 2024 | Two-stage latent compression → efficient FM training |

---

## Theme 4: Distillation and Few-Step Sampling

*Getting from 50 steps to 1 step without retraining.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| 16 | **Consistency Models** | Song et al. | ICML 2023 | Self-consistent ODE trajectories → 1-step generation (also applies to FM) |
| 17 | **Flow Consistency Models** | — | 2024 | Apply consistency distillation to flow matching models |
| 18 | **Shortcut Models: Your DiffusionModel is Secretly a One-Step Generator** | Frans et al. | 2024 | Learn to take large steps by predicting accumulated velocity |
| 19 | **Distilling the Knowledge of Diffusion with Flow Matching** | — | 2024 | Distill DDPM into a flow matching student |
| 20 | **SDXL-Turbo / ADD** | Sauer et al. | 2023 | Adversarial distillation → 1-step Stable Diffusion |

---

## Theme 5: Flow Matching Beyond Images

*Audio, molecules, proteins, and scientific applications.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| 21 | **Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale** | Le et al. (Meta) | NeurIPS 2023 | Flow matching for audio — in-context speech generation |
| 22 | **Audiobox: Unified Audio Generation with Natural Language Prompts** | Vyas et al. (Meta) | 2023 | Extends Voicebox to sound effects + music |
| 23 | **Flow Matching for Protein Structure Prediction** | — | 2024 | FM applied to AlphaFold-style structure prediction |
| 24 | **SE(3)-Stochastic Flow Matching for Protein Backbone Generation** | Bose et al. | ICML 2024 | FM on SE(3) manifold — the FM version of FrameDiff |
| 25 | **Protein Design with Guided Discrete Diffusion** | — | 2024 | Sequence + structure joint FM for protein design |
| 26 | **FoldFlow: De Novo Protein Structure Generation with Flow Matching** | Bose et al. | 2023 | Earlier FM for protein backbones — comparison to FrameDiff |
| 27 | **FrameDiff: Diffusion on SE(3) for Protein Backbone Generation** | Yim et al. | ICML 2023 | The cleaner alternative to RFdiffusion — preview of TLH-4 |
| 28 | **E3 Diffusion for Molecule Generation in 3D** | Hoogeboom et al. | NeurIPS 2022 | EDM — equivariant diffusion for molecules — key TLH-4 paper |
| 29 | **Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges** | Bronstein et al. | 2021 | The theoretical foundation for equivariant models on geometry |
| 30 | **Score-Based Generative Modeling through SDEs** | Song et al. | ICLR 2021 | The SDE unification — helps see where FM fits in the landscape |

---

## Suggested Reading Order

**Before the session (~4 hrs, ⭐ only):**
1. **Rectified Flow** (#2) — 60 min — start here: simplest FM formulation, direct connection to DDPM/DDIM
2. **Flow Matching for Generative Modeling / CFM** (#1) — 90 min — conditional FM, OT paths
3. **SD3** (#11) — 45 min — how this is applied in production: MM-DiT + FM + T5
4. **OT-FM** (#6) — 30 min — why pairing matters for path straightness

**To go deeper:**
- Theory: Albergo & Vanden-Eijnden (#3) for the unification with stochastic interpolants
- Distillation: Consistency Models (#16) → Flow Consistency Models (#17) for 1-step generation
- Biology: SE(3)-SFM (#24) + FoldFlow (#26) as the FM answer to RFdiffusion

**TLH-3 prep:**
- SiT (#13) — DiT + flow matching combined = the state-of-the-art setup
