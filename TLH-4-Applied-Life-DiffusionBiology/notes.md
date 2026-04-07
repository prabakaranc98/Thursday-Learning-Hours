# Session Notes — TLH-4: Applied Life: Diffusion for Biology & Molecular Dynamics

> **Date:** May 1, 2026 · **Format:** Foundations → Frontiers → Demo → Discussion

---

## 🧠 Core Thesis

> *"Protein design is a generative modeling problem over SE(3) — the group of rotations and translations in 3D. A protein backbone is a sequence of rigid frames, each a rotation R ∈ SO(3) and a translation t ∈ ℝ³. DDPM generalizes: define a forward process that adds noise on the SO(3) manifold (IGSO(3) distribution) and in ℝ³ (Gaussian). Train a neural network to reverse it. The result is RFdiffusion — a model that generates protein backbones conditioned on functional constraints."*

---

## 🏛️ FOUNDATIONS

### Section 1 — Why Proteins Need Special Treatment

**Protein backbone geometry:**
- A protein of $N$ residues is a sequence of $N$ **rigid frames** $(R_i, \mathbf{t}_i) \in SE(3)$
- $R_i \in SO(3)$ — orientation of backbone atoms N, Cα, C at residue $i$
- $\mathbf{t}_i \in \mathbb{R}^3$ — Cα position

**Requirements for a generative model:**
1. **Equivariance:** Rotating the whole protein by $g \in SE(3)$ gives a rotated version of the same protein — the model must respect this symmetry
2. **Validity:** Generated backbones must have valid bond lengths/angles
3. **Conditioning:** Generate backbones that bind to a specific target (drug design), fold into a specific topology, etc.

Standard image diffusion violates these requirements: it operates on pixel grids, not geometric objects.

---

### Section 2 — Diffusion on Manifolds

**Standard diffusion:** forward process on $\mathbb{R}^n$ — Gaussian noise.

**Manifold diffusion:** the data lives on a manifold $\mathcal{M}$ (e.g., $SO(3)$, $\mathbb{S}^2$, protein conformation space). We need:
- A forward process that stays on the manifold: $q(\mathbf{x}_t | \mathbf{x}_0)$ with $\mathbf{x}_t \in \mathcal{M}$
- A reverse process parameterized by an equivariant neural network

**IGSO(3) — Isotropic Gaussian on SO(3):**

The natural diffusion process on $SO(3)$ is:
$$q(R_t | R_0) = \mathrm{IGSO}(3)(R_t; R_0, \sigma_t)$$

As $\sigma_t \to \infty$: $R_t$ converges to the uniform distribution on $SO(3)$ (Haar measure).

The score function on $SO(3)$:
$$\nabla_{R_t} \log q(R_t | R_0) \in \mathfrak{so}(3) \quad \text{(Lie algebra of SO(3))}$$

---

### Section 3 — RFdiffusion (Watson et al., Nature 2022)

**Key architecture:** builds on RoseTTAFold (protein structure predictor), extended to act as a denoiser.

**Forward process:**
- Translation: $\mathbf{t}_t = \mathbf{t}_0 + \sigma_t\boldsymbol\epsilon$, $\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- Rotation: $R_t = R_0 \cdot \exp(\sigma_t \cdot \boldsymbol\omega)$, $\boldsymbol\omega \sim \mathrm{IGSO}(3)(0, 1)$

**Reverse process:** SE(3)-equivariant network predicts the score (noise direction) at each step.

**Conditioning:**
- Partial diffusion: fix some residues, diffuse others → scaffold design around functional motifs
- Sequence conditioning: generate backbones compatible with a given sequence
- Pocket conditioning: generate binders for a protein target

```
RFdiffusion pipeline:
x_T ~ IGSO(3) ⊗ N(0,I)    (random frames)
    ↓  T reverse steps
    ↓  RoseTTAFold denoiser (SE(3)-equivariant)
x_0 : designed backbone    (valid protein geometry)
    ↓  ProteinMPNN          (sequence design)
sequence + structure        → experimental validation
```

---

### Section 4 — FrameDiff (Yim et al. 2023)

An independent, cleaner derivation of protein backbone diffusion:

- **Explicit SO(3) diffusion** with closed-form forward marginals (similar to DDPM's closed-form marginal on ℝ)
- **Invariant Point Attention (IPA)** as the denoiser — borrowed from AlphaFold 2's structure module
- **Score prediction** in the Lie algebra $\mathfrak{se}(3)$

Training objective:
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0}\!\left[\|s_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0)\|^2\right]$$

where the score is computed in the tangent space of $SE(3)$ at $\mathbf{x}_t$.

---

### Section 5 — Equivariant Diffusion for Molecules (EDM)

Hoogeboom et al. (2022) — 3D molecule generation.

**Data:** atom positions $\mathbf{r}_i \in \mathbb{R}^3$ + atom types $h_i \in \mathbb{R}^d$ (one-hot).

**E(3)-equivariance requirement:**
$$f(g \cdot (\mathbf{r}, h)) = g \cdot f(\mathbf{r}, h) \quad \forall g \in E(3)$$

**Architecture: EGNN (E(3)-Equivariant Graph Neural Network):**
$$\mathbf{m}_{ij} = \phi_e(\mathbf{h}_i, \mathbf{h}_j, \|\mathbf{r}_i - \mathbf{r}_j\|^2)$$
$$\mathbf{r}_i \leftarrow \mathbf{r}_i + \sum_j \frac{\mathbf{r}_i - \mathbf{r}_j}{\|\mathbf{r}_i - \mathbf{r}_j\|}\,\phi_r(\mathbf{m}_{ij})$$
$$\mathbf{h}_i \leftarrow \phi_h(\mathbf{h}_i, \sum_j \mathbf{m}_{ij})$$

Key: the position update is a linear combination of **displacement vectors** weighted by scalar messages → equivariant by construction.

**Training:** DDPM over both positions $\mathbf{r}_t$ and features $h_t$:
$$\mathcal{L} = \mathbb{E}\!\left[\|\boldsymbol\epsilon_\mathbf{r} - \hat{\boldsymbol\epsilon}_\mathbf{r}\|^2 + \|\boldsymbol\epsilon_h - \hat{\boldsymbol\epsilon}_h\|^2\right]$$

---

### Section 6 — Molecular Dynamics as Diffusion (MDGen)

Molecular dynamics (MD) simulates the trajectory of a protein in time: $\{R_1, R_2, \ldots, R_T\}$ — a sequence of conformations.

**MDGen insight:** treat the MD trajectory as a generative modeling problem:
$$p(R_{1:T}) \approx p_\theta(R_{1:T})$$

Use a diffusion model over the **trajectory** (all conformations jointly) rather than per-frame:
- Forward process: add noise to all frames simultaneously
- Reverse process: denoise to recover a valid trajectory

**Why this works:** MD trajectories have strong inter-frame correlations (nearby frames are similar). A diffusion model can learn these correlations as part of its score function.

**Applications:**
- Generate conformational ensembles much faster than MD simulation
- Sample rare events (folding, binding) that MD would take ms/μs to reach
- Condition on endpoint conformations (e.g., bound vs unbound states)

---

### Section 7 — Structure-Based Drug Design (DiffSBDD)

Schneuing et al. (2022) — generate ligands conditioned on protein pocket geometry.

```
Input:  protein pocket P (3D atomic coordinates of binding site)
Output: small molecule ligand L ∈ pocket P

Model:  x_T (random 3D point cloud of atoms)
        → T reverse diffusion steps with P as conditioning context
        → x_0 (valid ligand geometry + atom types)
```

**Joint diffusion:** positions + atom types + bond types all diffuse simultaneously.

**Cross-attention conditioning:** pocket atoms act as "key/value" tokens; ligand atoms are "query" tokens — the ligand attends to the pocket at each denoising step.

---

## 🛠️ FRAMEWORK: TOY DIFFUSION ON MOLECULAR DATA

### Demo 01: 2D Point Cloud Diffusion

Simulate a toy "molecule" as $N$ particles in $\mathbb{R}^2$ with distance constraints. Apply standard Gaussian diffusion and an equivariant update rule:

```python
# Toy equivariant denoiser — 2D version of EGNN
class EGNNLayer2D(nn.Module):
    def forward(self, pos, feat, edges):
        # Pairwise displacements and distances
        diff = pos[edges[:, 0]] - pos[edges[:, 1]]     # (E, 2)
        dist = diff.norm(dim=-1, keepdim=True)           # (E, 1)
        # Message passing (invariant messages)
        msg = self.edge_mlp(torch.cat([feat[edges[:, 0]], feat[edges[:, 1]], dist**2], dim=-1))
        # Equivariant position update (displacement-weighted)
        pos_update = scatter(diff / (dist + 1e-8) * self.pos_mlp(msg),
                             edges[:, 0], dim=0, reduce='sum')
        pos = pos + pos_update
        feat = feat + scatter(msg, edges[:, 0], dim=0, reduce='sum')
        return pos, feat
```

### Demo 02: 1D Torsion Angle Diffusion

Protein backbone geometry is equivalent to a sequence of dihedral angles $(\phi_i, \psi_i) \in [-\pi, \pi]^2$.

Diffuse over the torus $\mathbb{T}^{2N}$:
- **Forward:** add von Mises noise $\phi_t = \phi_0 + \sigma_t \epsilon$, wrap to $[-\pi, \pi]$
- **Reverse:** 1D CNN or Transformer over the sequence of angle pairs
- **Output:** Ramachandran-valid backbone angles → reconstruct 3D backbone

---

## 💡 Key Takeaways

1. **Diffusion generalizes to any manifold.** The forward/reverse process framework extends beyond ℝⁿ to SE(3), SO(3), the torus, etc. — the geometry changes the forward process and the score computation, not the framework.

2. **Equivariance is not optional for molecular generation.** A model that is not SE(3)-equivariant will generate different molecules for differently-oriented inputs — physically nonsensical.

3. **RFdiffusion = AlphaFold + DDPM.** The denoiser is AlphaFold's structure prediction backbone; the training objective is score matching on SO(3)×ℝ³.

4. **Molecular dynamics + diffusion = fast conformational sampling.** MD is expensive; a diffusion model trained on MD trajectories can generate conformational ensembles in milliseconds.

5. **Drug design = conditional generation.** The pocket defines the conditioning context; the ligand is the generated output — same as text-conditional image generation but in 3D.

---

## ❓ Discussion Questions

- What is the difference between the IGSO(3) diffusion process and a standard Gaussian process on ℝ³? Why does this distinction matter for protein backbone generation?
- How does invariant point attention (IPA) achieve SE(3)-equivariance? How does this differ from standard multi-head attention?
- RFdiffusion generates backbones; ProteinMPNN then designs sequences. Why not do both jointly? What are the trade-offs?
- How would you condition a diffusion model to generate a protein that binds to a specific small molecule (the reverse of DiffSBDD)?
- MDGen generates trajectories — what does the "noise" look like in the conformational ensemble space? What defines "distance" between protein conformations?
