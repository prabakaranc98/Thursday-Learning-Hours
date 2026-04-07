# Slides Outline — TLH-4: Diffusion for Biology & Molecular Dynamics

> **19 slides · 60-minute session**
> Foundations (15 min) → Frontiers (15 min) → Demo (20 min) → Discussion (10 min)

---

## Slide 1 — Cover
**Title:** Diffusion for Biology & Molecular Dynamics
**Subtitle:** From pixel noise to protein frames — diffusion over SE(3) and conformational space
**Session:** TLH-4 · Thursday Learning Hours · Applied Life Track
*Visual: Protein backbone ribbon diagram with noise overlaid → clean designed structure*

---

## 🏛️ FOUNDATIONS (15 min, Slides 2–6)

## Slide 2 — From Pixels to Proteins
**Heading:** Same Framework, Different Space

DDPM on images: $\mathbf{x}_0 \in \mathbb{R}^{H \times W \times C}$ — a pixel grid

RFdiffusion on proteins: $\mathbf{x}_0 = \{(R_i, \mathbf{t}_i)\}_{i=1}^N$ — a sequence of SE(3) frames

**What changes:**
- Forward process: Gaussian noise on $\mathbb{R}^n$ → IGSO(3) noise on rotation manifold
- Denoiser: U-Net → SE(3)-equivariant GNN / IPA Transformer
- Loss: MSE on $\mathbb{R}^n$ → score matching on $SO(3) \times \mathbb{R}^3$

**What stays the same:**
- Training objective: denoising MSE (score matching)
- Sampling: 200-step DDPM reverse
- Architecture pattern: encode → denoise → decode

---

## Slide 3 — Protein Geometry 101
**Heading:** A Protein Backbone = A Sequence of SE(3) Frames

*Visual: Protein backbone with labeled backbone atoms N, Cα, C at one residue*

For each residue $i$:
- $R_i \in SO(3)$: the rotation matrix defining the local frame (N → Cα → C orientation)
- $\mathbf{t}_i \in \mathbb{R}^3$: the Cα position

Together: $(R_i, \mathbf{t}_i) \in SE(3)$ — a rigid frame in 3D space.

**Why SE(3) and not $\mathbb{R}^{9+3}$?**
Rotations live on a manifold — adding noise in the parameterization space $(\mathbb{R}^9)$ doesn't give uniform noise over orientations. We need geometry-aware noise.

---

## Slide 4 — Diffusion on SO(3)
**Heading:** IGSO(3) — The Gaussian of Rotation Space

**Isotropic Gaussian on SO(3):** $\mathrm{IGSO}(3)(R; R_0, \sigma)$

- At $\sigma \to 0$: concentrated at $R_0$
- At $\sigma \to \infty$: uniform over $SO(3)$ — maximum entropy (Haar measure)

**Closed-form forward process:**
$$q(R_t | R_0) = \mathrm{IGSO}(3)(R_t;\; R_0,\; \sigma_t)$$

implemented as: $R_t = R_0 \cdot \exp(\sigma_t \cdot \boldsymbol\omega)$, where $\boldsymbol\omega \in \mathfrak{so}(3)$ is a random axis-angle.

**Score in Lie algebra:**
$$\nabla_{R_t} \log q(R_t | R_0) \in \mathfrak{so}(3) \cong \mathbb{R}^3$$

*Visual: Points on a sphere being displaced — analogue of Gaussian noise on a 2-sphere*

---

## Slide 5 — Equivariance: The Non-Negotiable Constraint
**Heading:** Why E(3)-Equivariance Is Essential

**The problem:** if you rotate all protein atoms by $g \in E(3)$, the output should also rotate by $g$.

Formally: $f(g \cdot \mathbf{x}) = g \cdot f(\mathbf{x})$

**Non-equivariant models fail:**
- A standard CNN/Transformer will predict different structures for the same protein in different orientations
- Training requires all proteins to be pre-aligned → loses generality

**E(3)-equivariant architectures:**
- **EGNN:** message passing with displacement vectors — simple and effective
- **SE(3)-Transformer:** spherical harmonic basis — theoretically clean
- **IPA (AlphaFold 2):** Invariant Point Attention — the denoiser in FrameDiff

---

## Slide 6 — EGNN: Equivariant GNN
**Heading:** Simple Equivariance via Displacement Vectors

```
For atoms i, j:
  diff_ij = pos_i - pos_j                           # displacement (equivariant)
  dist_ij = ||diff_ij||                              # distance (invariant)
  
  msg_ij  = MLP(feat_i, feat_j, dist_ij²)           # invariant message
  
  # Equivariant position update
  pos_i  += Σ_j  (diff_ij / dist_ij) * MLP(msg_ij)  # weighted sum of displacements
  feat_i += Σ_j  msg_ij                               # invariant feature update
```

The key: position updates are weighted sums of **displacement vectors** — automatically equivariant under rotation/translation.

*Visual: 2D molecule with arrows showing equivariant displacement-weighted updates*

---

## 🔭 FRONTIERS (15 min, Slides 7–11)

## Slide 7 — RFdiffusion: Protein Design at Scale
**Heading:** Watson et al., Nature 2023 — Experimentally Validated Protein Design

**Architecture:** RoseTTAFold (protein structure predictor) used as SE(3)-equivariant denoiser.

**What it can design:**
- Monomers: arbitrary topology de novo proteins
- Binders: proteins that bind to a target protein surface
- Symmetric assemblies: icosahedral cages, helical filaments
- Motif scaffolding: build a scaffold around a functional site (enzyme active site, binding epitope)

**Results:**
- 8 of 25 de novo binders validated experimentally in the first paper
- Sub-nanomolar affinities achieved without further optimization

*Visual: RFdiffusion pipeline — random frames → denoising → backbone → ProteinMPNN → sequence → AlphaFold validation*

---

## Slide 8 — FrameDiff: Cleaner Theory
**Heading:** SE(3) Diffusion with IPA Denoiser

Yim et al. (ICML 2023) — independent, cleaner derivation:

1. **Closed-form marginals** on SE(3): same as DDPM's closed-form marginal on ℝ (Section 4 of TLH-1)
2. **IPA denoiser:** AlphaFold 2's Invariant Point Attention — SE(3)-equivariant by construction
3. **Score prediction** in the Lie algebra $\mathfrak{se}(3)$ — analogous to $\boldsymbol\epsilon$ in DDPM

**Training objective:**
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0}\!\left[\|s_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0)\|^2_{\mathfrak{se}(3)}\right]$$

*The DDPM MSE loss, but the norm is in the Lie algebra of SE(3).*

---

## Slide 9 — EDM: Molecules in 3D
**Heading:** E(3)-Equivariant Diffusion for Drug-Like Molecules

Hoogeboom et al. (ICML 2022) — generate 3D molecular graphs.

**Data:** $(\mathbf{r}_i \in \mathbb{R}^3, h_i \in \mathbb{R}^K)$ — atom positions + atom type features

**Forward process:**
- Positions: $\mathbf{r}_t = \sqrt{\bar\alpha_t}\mathbf{r}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$ — standard Gaussian (E(3)-equivariant if we subtract center of mass first)
- Features: $h_t$ — same DDPM process on the feature dimension

**Denoiser:** EGNN — equivariant GNN processes all atoms jointly with pairwise interactions.

**Results:** State-of-the-art on QM9 and GEOM datasets — stable, valid, drug-like molecules.

---

## Slide 10 — DiffSBDD: Structure-Based Drug Design
**Heading:** Pocket-Conditioned Ligand Generation

Schneuing et al. — generate small molecules conditioned on a protein binding pocket.

```
Input:  protein pocket P — 3D atomic coordinates of binding residues
Output: small molecule ligand L — positioned and shaped to fit pocket P

Architecture:
  Pocket atoms    → key/value tokens  ─────────────────┐
  Ligand atoms    → query tokens      → cross-attention ┘
  → equivariant update of ligand positions + atom types
  → repeat T times (denoising steps)
```

**The binding problem as conditional generation:** same as text-to-image, but:
- "Text" = protein pocket geometry
- "Image" = 3D molecule + bonds

*Visual: Protein pocket surface with generated ligand fitting snugly inside*

---

## Slide 11 — MDGen: Molecular Dynamics via Diffusion
**Heading:** Generate Conformational Ensembles Without Running MD

**Problem:** MD simulation of a protein takes milliseconds (real time) per microsecond of dynamics.

**MDGen approach:** learn $p(\{R_1, ..., R_T\})$ — the joint distribution of conformations in a trajectory.

```
Input:  protein sequence + target ensemble length T
Output: T conformations sampled from equilibrium ensemble

Training: fit diffusion model on MD simulation trajectories
Sampling: 20ms per ensemble (vs hours for MD)
```

**Connection to video diffusion (TLH-3):** an MD trajectory is literally a video — frames of a protein's conformation evolving in time. The same temporal attention machinery applies.

*Visual: RMS deviation plot — MDGen ensemble (ms) matches MD ensemble (hours)*

---

## 🛠️ DEMO (20 min, Slides 12–16)

## Slide 12 — What We Build
**Heading:** Demo Goal: Toy Equivariant Diffusion + 1D Torsion Angle Diffusion

```
Demo 01 — 2D Molecular Diffusion:
  ✓ N particles in ℝ² with distance constraints (toy molecule)
  ✓ Standard Gaussian forward process (subtract center of mass → E(2)-invariant)
  ✓ Toy EGNN denoiser: pairwise distances → displacement-weighted updates
  ✓ Visualise: particle cloud denoising back to valid geometry

Demo 02 — 1D Backbone Torsion Diffusion:
  ✓ Protein backbone ≈ sequence of (φ, ψ) torsion angles ∈ [-π, π]²
  ✓ Wrapped Gaussian diffusion on the torus 𝕋²
  ✓ 1D CNN or Transformer denoiser over the sequence
  ✓ Visualise: Ramachandran plot of generated angles
```

---

## Slide 13 — Live Demo: Part 1 — EGNN Denoiser
**Heading:** `demo_01_molecular_diffusion_2d.ipynb` — Equivariant Updates

```python
class EGNNLayer(nn.Module):
    def forward(self, pos, feat, edges):
        i, j = edges                           # source, target indices
        diff  = pos[i] - pos[j]               # displacement (equivariant)
        dist2 = (diff ** 2).sum(-1, keepdim=True)

        # Invariant message
        msg = self.edge_mlp(torch.cat([feat[i], feat[j], dist2], dim=-1))

        # Equivariant position update
        pos_delta = scatter(
            diff / (dist2.sqrt() + 1e-8) * self.pos_gate(msg),
            i, dim=0, dim_size=pos.shape[0], reduce='sum'
        )
        pos  = pos + pos_delta
        feat = feat + scatter(msg, i, dim=0, dim_size=feat.shape[0], reduce='sum')
        return pos, feat
```

*Show: toy molecule (6 atoms in ℝ²) denoising — positions converge to valid ring geometry*

---

## Slide 14 — Live Demo: Part 2 — Torsion Angle Diffusion
**Heading:** Ramachandran Space as a Diffusion Target

```python
# Wrapped Gaussian on [-π, π]
def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def q_sample_torsion(phi0, psi0, t, noise):
    sigma_t = t[:, None] * 1.5               # max noise = 1.5 rad
    phi_t   = wrap(phi0 + sigma_t * noise[:, :, 0])
    psi_t   = wrap(psi0 + sigma_t * noise[:, :, 1])
    return phi_t, psi_t

# Denoiser: 1D Transformer over residue sequence
class TorsionDenoiser(nn.Module):
    def forward(self, phi_t, psi_t, t):
        # embed angles as (sin, cos) pairs — natural periodic encoding
        x = torch.stack([phi_t.sin(), phi_t.cos(), psi_t.sin(), psi_t.cos()], dim=-1)
        ...
        return dphi, dpsi  # predicted score in torsion space
```

*Show: Ramachandran scatter plot — generated (φ, ψ) pairs cluster in α-helix and β-sheet regions*

---

## Slide 15 — Demo: Equivariance Verification
**Heading:** Show That the Model Is Actually Equivariant

```python
# Rotate all atoms by a random rotation matrix
R = random_rotation_matrix()
pos_rotated = pos @ R.T

# Equivariant model: output should rotate by same R
out_original = model(pos, feat, edges)
out_rotated  = model(pos_rotated, feat, edges)

# Check equivariance
diff = (out_rotated.pos - out_original.pos @ R.T).abs().max()
print(f'Equivariance error: {diff:.6f}')   # should be ~1e-6 (floating point)
```

*Show: equivariance error plot across rotations — flat at ~1e-6 = machine precision*

---

## Slide 16 — Demo: Connection to AlphaFold
**Heading:** AlphaFold 2's IPA = SE(3)-Equivariant Attention

Invariant Point Attention (IPA) — the attention module in AlphaFold 2's structure module:

```
For residue pair (i, j):
  q_i, k_j, v_j  in local frame of residue i
  attention weight = softmax(q_i · k_j / √d - ||T_i · q_i^pt - T_j · k_j^pt||²)
  output = Σ_j weight_ij * v_j  (in local frame of i)
```

- $T_i$ = rigid frame of residue $i$
- Point projections $q^{pt}, k^{pt} \in \mathbb{R}^3$ computed in local frames → invariant inner products
- Output features + updated positions → SE(3)-equivariant

FrameDiff uses this exact module as its denoiser. RFdiffusion uses RoseTTAFold's structure module — same principle.

---

## 🔄 DISCUSSION (10 min, Slides 17–19)

## Slide 17 — The Landscape
**Heading:** Diffusion Models Across Biological Data Types

| Data | Forward noise | Equivariance | Key paper |
|---|---|---|---|
| Protein backbone | IGSO(3) + ℝ³ Gaussian | SE(3) | RFdiffusion, FrameDiff |
| Small molecules | ℝ³ Gaussian (CoM-free) | E(3) | EDM |
| MD trajectories | ℝ³ Gaussian per frame | E(3) (per step) | MDGen |
| Protein sequences | Discrete (MDLM) | Permutation | ProteinMPNN + discrete FM |
| Antibody CDRs | SE(3) + discrete | SE(3) | AbDiffuser |
| RNA structure | SE(3) | SE(3) | — |

---

## Slide 18 — Open Questions

1. RFdiffusion generates backbone; ProteinMPNN designs sequences. Can you do both jointly in one diffusion model?
2. What is the "score function" in protein space? What does moving in the direction of $\nabla_{R} \log p(R)$ mean physically?
3. MDGen generates conformational ensembles. How do you validate that they match equilibrium thermodynamics?
4. DiffSBDD generates ligands in a pocket — but drug-like properties (ADMET) aren't encoded. How would you add these constraints?
5. AlphaFold 3 uses diffusion for all-atom prediction. Is there a fundamental reason diffusion (not AR, not regression) is the right approach for structure prediction?

---

## Slide 19 — Takeaways + TLH-6 Preview

**Today:**
- Diffusion = universal framework; change the manifold, change the forward process, keep the training objective
- SE(3)-equivariance is mandatory for protein/molecule generation — geometry must be respected
- RFdiffusion = AlphaFold backbone + IGSO(3) DDPM — validated in wet lab
- EDM = EGNN + DDPM for 3D molecule generation
- MDGen = diffusion over MD trajectories = fast conformational sampling

**TLH-6 — Applied Life: AI × Biology Survey:**
- Broader landscape: ESM-3, AIDO, Evo 2 — foundation models for biology
- Perturbation biology: GEARS, Perturb-seq, CRISPR as do-interventions
- Causal Representation Learning (CRL) × biology: do knockouts satisfy CRL identifiability conditions?
- The full AI × drug discovery pipeline: target → structure → ligand → ADMET → clinical

---

## Design Notes
- **Colors:** ivory `#f7f3ea`, copper `#8f3e22`, charcoal text
- **Protein visuals:** ribbon diagrams, Ramachandran plots, pocket surfaces
- **Math:** typeset with KaTeX / LaTeX, copper for key equations
