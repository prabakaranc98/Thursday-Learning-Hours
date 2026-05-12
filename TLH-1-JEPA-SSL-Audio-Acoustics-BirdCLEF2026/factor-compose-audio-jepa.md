# Factorize-and-Compose Audio-JEPA

## Problem Statement

BirdCLEF-style prediction is a multi-label soundscape classification problem.
The model receives short windows from long passive acoustic monitoring
recordings and must predict which species are present in each window.

The hard part is not just building an audio classifier. The hard part is
learning a representation that can handle:

- multiple species in the same soundscape window,
- weak and noisy labels,
- overlap between calls, background, insects, rain, wind, and recorder noise,
- domain shift between curated species recordings and passive field
  soundscapes,
- rare species where only a small number of clean examples may exist.

The central research question for this TLH is:

> Can we build a JEPA-style audio model whose predictor learns factored
> species/event latents, and whose latent space supports approximate additive
> composition when multiple species occur together?

The downstream task remains simple:

```text
soundscape window -> multi-label species probabilities
```

The learning problem is richer:

```text
soundscape window -> factored acoustic latents -> composed latent -> classifier
```

## Data Distinction

There are two different kinds of training signal.

### Train Soundscapes

Train soundscapes are closest to the test distribution. They contain long field
recordings with the same kinds of clutter and overlap that appear at inference
time.

They are useful for:

- target-domain acoustic structure,
- background and recorder conditions,
- unlabeled or weakly labeled self-supervised learning,
- direct multi-label classification if window-level labels are available.

### Train Audio

Train audio consists of species-specific recordings. These clips are better for
learning what each species can sound like, but they may be cleaner, shorter, or
more curated than the test soundscapes.

They are useful for:

- species identity,
- prototype or memory construction,
- supervised species discrimination,
- synthetic mixture construction.

This means the final task is not multi-instance learning by definition. The
final task is multi-label classification. MIL-like machinery may still be
useful when a long train audio file has only file-level labels and we do not
know which crop contains the labeled species.

## BirdCLEF 2026 Data Reality

The Kaggle task pages should be treated as the source of truth:

- Overview: <https://www.kaggle.com/competitions/birdclef-2026/overview>
- Data: <https://www.kaggle.com/competitions/birdclef-2026/data>

For the architecture, the important constraints are:

- prediction is over short windows from test soundscapes,
- each prediction is multi-label over the competition species list,
- `train_audio/` contains short species recordings from xeno-canto and
  iNaturalist,
- `train_soundscapes/` and `train_soundscapes_labels.csv` provide in-domain
  Pantanal soundscape examples,
- the hidden test set is populated during notebook scoring,
- CPU inference budget matters, so the final model cannot be arbitrarily
  heavy.

This strengthens the case for using both data sources. `train_audio` gives
species identity. `train_soundscapes` gives the target acoustic domain. A model
that ignores either one is structurally mismatched to the competition.

This also matters for species coverage. Some classes may be represented much
better in labeled soundscape windows than in curated `train_audio`; sonotypes
and non-bird taxa are especially important. The architecture must therefore be
able to learn from both individual recordings and labeled soundscape windows,
not only from focal species clips.

## How We Arrived Here

The initial baseline is straightforward:

```text
audio window -> log-mel spectrogram -> encoder -> sigmoid species head
```

This is necessary, but it does not use the two data sources in a principled
way. It treats the representation as one pooled vector and asks a classifier to
separate everything after the fact.

Standard JEPA improves the representation learning part:

```text
visible patches -> context encoder -> predictor -> target latent
masked patches  -> target encoder  -> target latent
```

But a conventional JEPA target is usually a single latent representation. For
BirdCLEF, a soundscape window may contain several species and background
events. A single latent can hide those sources in an entangled vector.

The next idea is to make the predictor produce factored latents:

```text
soundscape -> encoder -> predictor -> [slot_1, slot_2, ..., slot_K]
```

The slots should behave like species/event components. A simple classifier can
then read those components for multi-label prediction.

The final step is to add a composition constraint. If a soundscape contains
species A and B, the soundscape latent should be explainable by the component
latents for A, B, and background:

```text
latent(soundscape_A+B) ~= compose(latent(A), latent(B), latent(background))
```

This gives the proposed architecture:

```text
encoder -> factored predictor -> composition predictor -> composed latent
```

## Core Hypothesis

The model should learn a latent space where species and acoustic events are
approximately compositional.

For individual recordings:

```text
x_A -> target encoder -> z_A
x_B -> target encoder -> z_B
```

For a mixture or soundscape:

```text
x_mix(A, B, background) -> context encoder -> predictor -> z_mix_hat
```

The desired relationship is:

```text
z_mix_hat ~= compose(z_A, z_B, z_background)
```

The compose function can start as a normalized sum:

```text
z_target = normalize(z_A + z_B + z_background)
```

Later it can become a learned DeepSets-style or attention-based composition
module.

This should be treated as a soft latent-space constraint. Raw waveforms are
additive, but log-mel spectrograms and semantic latents are not exactly
additive.

## Proposed Architecture

Call the model **Factorize-and-Compose Audio-JEPA**.

```text
soundscape window
  -> log-mel spectrogram
  -> context encoder E_c
  -> acoustic evidence tokens
  -> predictor stage 1 P1: factored slots
  -> predictor stage 2 P2: composed mixture latent
  -> multi-label classifier over P1 slots
```

Target path:

```text
individual species audio
  -> log-mel spectrogram
  -> target encoder E_t
  -> species/component latent

masked or full soundscape target
  -> target encoder E_t
  -> mixture target latent
```

### Encoder

The encoder is the acoustic evidence extractor. It should preserve enough
time-frequency detail for later factorization.

Candidate encoder shapes:

- small CNN over log-mel spectrograms,
- CNN stem plus transformer blocks,
- AST-style patch transformer initialized from scratch,
- lightweight Conformer-style encoder initialized from scratch.

The build constraint is important: we can use these architectural ideas, but
the weights should start random. We should not start from BirdNET, Perch,
AudioSet, ImageNet, AudioMAE, or other pretrained checkpoints.

## Perch As Baseline And Foil

Perch is the strongest obvious comparison. Google describes Perch 2.0 as a
pre-trained bioacoustics model that provides off-the-shelf classification scores
and transfer-learning embeddings across many vocalizing species:
<https://research.google/pubs/perch-20-the-bittern-lesson-for-bioacoustics/>.
The Perch repository says the released model is available through Kaggle Models,
uses a PCEN mel-spectrogram frontend and EfficientNet model family, and supports
embedding/search/agile-modeling workflows:
<https://github.com/google-research/perch>.

The original Perch paper is also important. It reports Perch as an EfficientNet
B1 trained on weakly labeled Xeno-Canto bird recordings, with activity
detection, MixUp, random gain, time shifting, and taxonomic supervision:
<https://www.nature.com/articles/s41598-023-49989-z>. That paper argues that
global bird embeddings transfer strongly to downstream bioacoustic tasks.

So the honest comparison is:

```text
Perch:
  strong pretrained supervised bioacoustic embedding model
  optimized for transfer and agile modeling
  excellent baseline for few-shot species classifiers

Factorize-and-Compose Audio-JEPA:
  scratch-trained research architecture
  optimized for in-domain soundscape composition
  explicitly learns factored slots and additive mixture geometry
```

We should not claim the new model is better than Perch until it wins ablations.
Perch has the advantage of massive external pretraining. Our advantage, if it
exists, should come from a different inductive bias.

### Where Perch Is Strong

- Large-scale supervised pretraining on bioacoustic species.
- Strong embeddings for few-shot transfer.
- Practical tooling for embedding audio and building small classifiers.
- Efficient inference/export path through released model tooling.
- Proven value in bioacoustic transfer learning.

### Where Our Architecture Tries To Be Different

Perch mostly gives one embedding or classifier output per audio window. That is
very useful, but it does not directly enforce a latent decomposition of a
multi-species soundscape.

Factorize-and-Compose Audio-JEPA tries to impose additional structure:

- **factored slots:** expose multiple species/event components in one window,
- **positive composition:** represent mixtures as gated sums of component
  latents,
- **soundscape grounding:** pretrain and fine-tune directly on target-domain
  Pantanal soundscapes,
- **dual-source training:** use `train_audio` for component identity and
  `train_soundscapes` for natural mixtures,
- **classifier simplicity:** make a shallow multi-label head work on P1 slots,
- **interpretability:** inspect which slot matched which prototype,
- **scratch-first learning:** test whether the algorithmic bias works without
  importing a foundation model.

### How To Compare Against Perch

Perch should become an evaluation baseline, not part of the first model:

```text
Baseline A:
  scratch CNN/transformer encoder + BCE

Baseline B:
  Perch embeddings + shallow multi-label classifier

Baseline C:
  Perch embeddings + prototype/similarity classifier

Our model:
  Factorize-and-Compose Audio-JEPA + shallow classifier on P1 slots
```

Compare on:

- macro ROC-AUC over soundscape windows,
- rare species and non-bird taxa,
- species that are better represented in soundscape labels than train audio,
- synthetic mixture slot recovery,
- nearest-prototype slot purity,
- CPU inference time,
- calibration and threshold stability.

The strongest possible outcome is not only:

```text
our AUC > Perch AUC
```

It is also:

```text
our slots explain multi-species mixtures better
our model adapts to Pantanal soundscapes with less external pretraining
our classifier is simpler because P1 already factorizes the evidence
```

If Perch wins on raw AUC, the architecture can still be scientifically useful
if it gives better decomposition, rare-class behavior, or in-domain adaptation.

### Target Encoder

Use standard JEPA mechanics:

```text
E_c = trainable context encoder
E_t = EMA copy of E_c
```

The target encoder should use stop-gradient. EMA keeps the target space more
stable and reduces the risk of the context path chasing its own moving outputs.

### Predictor Stage 1: Factored Latents

P1 maps encoder tokens into a fixed number of latent slots:

```text
P1(E_c(x)) = [s_1, s_2, ..., s_K]
```

Each slot should represent one component of the soundscape: species call,
background event, silence/noise, or another acoustic source.

The slots should not be directly ordered. Slot 1 does not have to mean a
specific species. Matching should be set-based.

Possible P1 designs:

- learned slot queries attending over encoder tokens,
- Slot Attention,
- transformer decoder with learned queries,
- modern Hopfield retrieval layer over acoustic tokens.

### Predictor Stage 2: Composition Latent

P2 takes the factored slots and composes them into one mixture-level latent:

```text
P2([s_1, ..., s_K]) = z_mix_hat
```

P2 exists as a training constraint. It forces the factored slots to still
explain the whole soundscape. Without this, P1 may learn slots that only help
classification but do not form a meaningful representation of the soundscape.

Possible P2 designs:

- sum or mean pooling plus MLP,
- attention pooling over slots,
- DeepSets-style composition,
- transformer over slots followed by pooling.

### Multi-Label Classifier

The classifier should read P1 slots, not only the pooled encoder output:

```text
[s_1, ..., s_K] -> species logits -> sigmoid probabilities
```

This is the main downstream path. The classifier can be shallow because the
factorization should carry most of the burden.

We should still ablate:

```text
encoder pooled tokens -> classifier
P1 slots              -> classifier
P2 composed latent    -> classifier
encoder + P1 slots    -> classifier
```

My expectation is that P1 slots will be most useful for multi-label prediction,
because they are explicitly trained to expose components.

## Training Modes

The model should use both masked and unmasked training modes.

### Mode 1: Masked Soundscape JEPA

Purpose: learn target-domain acoustic structure from soundscapes.

```text
visible soundscape patches -> E_c -> P1 -> P2 -> z_hat_target
masked target patches      -> E_t             -> z_target
```

Loss:

```text
L_jepa = distance(z_hat_target, stopgrad(z_target))
```

This is closest to conventional JEPA.

### Mode 2: Unmasked Composition From Known Components

Purpose: teach additive/compositional structure.

Construct synthetic examples:

```text
x_mix = mix(x_A, x_B, background)
```

Target components:

```text
z_A  = E_t(x_A)
z_B  = E_t(x_B)
z_bg = E_t(background)
```

Predicted factors:

```text
[s_1, ..., s_K] = P1(E_c(x_mix))
z_mix_hat      = P2([s_1, ..., s_K])
```

Composed target:

```text
z_comp_target = normalize(z_A + z_B + z_bg)
```

Losses:

```text
L_factored = set_match([s_1, ..., s_K], [z_A, z_B, z_bg])
L_composed = distance(z_mix_hat, stopgrad(z_comp_target))
```

This mode is the core of the new idea.

### Mode 3: Natural Labeled Soundscape Training

Purpose: connect factored latents to the actual BirdCLEF prediction task.

```text
soundscape window -> E_c -> P1 -> classifier -> multi-label species logits
```

Loss:

```text
L_cls = BCEWithLogitsLoss(logits, multi_hot_labels)
```

If a window label says species A and C are present, we can also retrieve
species prototypes from train audio and use them as weak factor targets:

```text
target set = [prototype_A, prototype_C, background_prototype]
```

This target is imperfect because the exact call in the soundscape may differ
from the curated species clips. Use it as a soft auxiliary loss, not the only
training signal.

### Mode 4: Train Audio Species Learning

Purpose: learn species identity from individual recordings.

For short species clips:

```text
clip -> E_c -> classifier -> species label
```

For long weakly labeled train audio:

```text
clip -> crops -> E_c per crop -> P1 per crop -> pooled prediction
```

This is where MIL-like pooling may be useful. It is a training technique for
weak file-level labels, not the definition of the final BirdCLEF task.

### Mode 5: Prototype Or Memory Refresh

Purpose: build target component latents from train audio.

For each species, maintain multiple prototypes:

```text
species c -> [p_c1, p_c2, ..., p_cM]
```

Multiple prototypes are better than one because a species can have different
calls, recording qualities, distances, and noise conditions.

The prototype pool can be:

- fixed after a warmup phase,
- periodically recomputed with the EMA target encoder,
- implemented as a learnable memory bank updated by training.

## Representation Geometry

The architecture only makes sense if the latent geometry is designed
deliberately. We do not want arbitrary features that happen to classify well.
We want a space where species/event components are separable, composable, and
usable by a shallow multi-label classifier.

### Unit-Sphere Components

Start with normalized latent vectors:

```text
z = normalize(E_t(x))
s_i = normalize(P1_i(E_c(x)))
```

Then cosine distance becomes the main comparison:

```text
distance(a, b) = 1 - cosine(a, b)
```

This makes prototype matching and set matching stable. It also prevents the
model from satisfying losses only by changing vector norms.

### Components Plus Gates

P1 should output both component directions and presence gates:

```text
P1(E_c(x)) = [(s_1, a_1), ..., (s_K, a_K)]
```

Where:

```text
s_i = slot direction
a_i = non-negative gate or confidence
```

The direction says what acoustic component the slot represents. The gate says
how strongly that component is present. This is important because mixture
composition needs both identity and strength.

For downstream classification:

```text
species logits = classifier([(s_i, a_i)])
```

For composition:

```text
z_composed = normalize(sum_i a_i * s_i)
```

In practice, P2 can be a learned composition module, but the simple gated sum is
the geometric reference point.

### Positive Cone View

A soundscape window should live near the positive cone spanned by the components
that are present:

```text
z_soundscape ~= normalize(
  alpha_A * z_A
+ alpha_B * z_B
+ alpha_bg * z_bg
)
```

The coefficients should be non-negative. This matters: a species should be
added when present, not subtracted. This also makes the latent space easier to
interpret.

This is not exact physics. It is a representation constraint. Waveforms are
additive, log-mel features are only approximately additive, and semantic
embeddings are softer still.

### Triangle Consistency

For synthetic mixtures, we can form a triangle of targets:

```text
individual components:
  z_A, z_B, z_bg

composed component target:
  z_comp = normalize(z_A + z_B + z_bg)

mixture target:
  z_mix_target = E_t(mix(A, B, bg))

prediction:
  z_mix_hat = P2(P1(E_c(mix(A, B, bg))))
```

The geometry should make these three points agree:

```text
z_mix_hat    close to z_comp
z_mix_hat    close to z_mix_target
z_mix_target close to z_comp
```

This gives a stronger constraint than only matching the target encoder output.
It says the predicted mixture latent must be compatible with both the natural
mixture and the sum of the individual component latents.

### Set Geometry For Slots

Slot order should not matter:

```text
[slot_1, slot_2, slot_3] == [slot_3, slot_1, slot_2]
```

So factor matching should be set-based:

```text
L_factored =
  min_assignment sum_j distance(predicted_slot_assignment(j), target_component_j)
```

For synthetic mixtures, the target set is known exactly:

```text
target set = [z_A, z_B, z_bg]
```

For natural soundscapes, the target set is weaker:

```text
target set = species prototypes from labels + background prototypes
```

Use the natural-soundscape factor loss with lower weight because the prototype
may not match the exact call in that soundscape.

### Class Geometry

Each species should become a region or collection of prototypes, not one single
point:

```text
species c -> [p_c1, p_c2, ..., p_cM]
```

The reason is acoustic variation:

- different call types,
- distance from recorder,
- sex/age variation,
- background and weather,
- overlapping species,
- recording quality.

The classifier can be partly similarity-based:

```text
logit_c =
  max_i,k temperature * cosine(s_i, p_ck)
  + linear_residual_c([s_1, ..., s_K])
```

This makes the classifier less mysterious. A high score for species c should
come from at least one slot being near one of that species' prototypes.

### Margins

For synthetic mixtures and trusted labels, use a margin objective:

```text
positive species:
  slot should be close to at least one positive prototype

negative species:
  slot should be farther than margin from negative prototypes
```

A simple form:

```text
L_margin =
  max(0, margin + sim(slot, negative_proto) - sim(slot, positive_proto))
```

Be careful with natural soundscape negatives. If labels are incomplete, an
absent label may not be a true negative. Strong negative margins are safest on
synthetic mixtures where we control the components.

### Slot Diversity Without Over-Forcing Orthogonality

Slots should not all collapse to the same vector. A basic diversity penalty is:

```text
L_diverse = mean_{i != j} max(0, cosine(s_i, s_j) - rho)
```

This discourages duplicate slots while allowing related species to remain
nearby. Do not force all slots to be orthogonal. Similar species and similar
background events may legitimately share acoustic structure.

### Sparsity And Empty Slots

Most windows contain only a few salient sources. The slot gates should be
sparse:

```text
L_sparse = sum_i a_i
```

But the model also needs a way to represent silence, background, and unused
capacity. Add explicit empty/background behavior:

```text
empty slots -> low gate or background prototype
```

For synthetic mixtures, the expected number of active slots is known. We can
regularize:

```text
sum_i active(a_i) ~= number_of_components_in_mix
```

For natural soundscapes, keep this weaker because the true number of events is
unknown.

### Background Geometry

Background cannot be a trash bucket that absorbs rare species. It should have
its own controlled geometry:

```text
background prototypes = [p_bg1, p_bg2, ..., p_bgM]
```

A slot can match background only if it is closer to background prototypes than
to species prototypes, or if its gate is low. This gives rare species a better
chance to occupy their own slot.

Useful regularizers:

```text
species slots should not be too close to background prototypes
background slots should have low species logits
rare species examples should receive higher matching weight
```

### Batch-Level Anti-Collapse

EMA targets and stop-gradient help, but slot models can still collapse. Add a
batch-level variance/covariance regularizer on active slots:

```text
variance:
  each latent dimension should have non-trivial variance across the batch

covariance:
  different dimensions should not become perfectly redundant
```

This is similar in spirit to VICReg or Barlow Twins, but applied to active
component slots rather than one global embedding.

### Desired Invariances And Equivariances

The representation should encode these properties:

```text
time shift:
  same call shifted slightly in time should keep similar species slot

mixing commutativity:
  compose(A, B) ~= compose(B, A)

additive equivariance:
  adding species B to A should add a B-like component slot

background robustness:
  species slot should remain stable across different backgrounds

slot permutation:
  the set of slots matters, not their order
```

These are the real geometric assumptions behind the architecture.

### Geometry Probes

We should validate the geometry directly, not only with leaderboard metrics.

Useful probes:

- nearest-prototype retrieval for each predicted slot,
- slot purity on synthetic mixtures,
- whether `latent(A+B)` is closer to `compose(A, B)` than to unrelated
  compositions,
- whether the same species lands in similar slots across backgrounds,
- whether rare species are separated from background slots,
- whether P1 slots outperform pooled encoder features for multi-label
  classification.

## Loss Summary

The full objective can be:

```text
L_total =
  lambda_jepa     * L_jepa
+ lambda_factored * L_factored
+ lambda_composed * L_composed
+ lambda_cls      * L_cls
+ lambda_diverse  * L_slot_diversity
+ lambda_empty    * L_empty_slot
+ lambda_sparse   * L_gate_sparsity
+ lambda_margin   * L_class_margin
+ lambda_varcov   * L_batch_anti_collapse
```

Where:

```text
L_jepa:
  predicts masked target latents from visible soundscape context

L_factored:
  matches predicted slots to component latents from individual audio

L_composed:
  matches P2's composed latent to the sum or learned composition of component
  target latents

L_cls:
  trains multi-label species prediction

L_slot_diversity:
  discourages all slots from collapsing to the same component

L_empty_slot:
  allows unused slots to represent silence/background/none instead of forcing
  every slot to match a species

L_gate_sparsity:
  encourages only a small number of active slots per window

L_class_margin:
  pushes trusted positive species slots closer to their prototypes than to
  negative species prototypes

L_batch_anti_collapse:
  keeps active slot dimensions from collapsing across a batch
```

Set matching can use:

- Hungarian matching over cosine distances,
- optimal transport,
- attention-based soft assignment.

## Downstream Usage

In standard JEPA, downstream usually uses the encoder and discards the
predictor. In this architecture, the predictor is not disposable.

The intended downstream representation is:

```text
P1 factored slots
```

Reason:

- the downstream task is multi-label,
- the prediction needs species-level separation,
- P1 is explicitly trained to expose components,
- P2 only verifies that the components still explain the whole mixture.

The simplest downstream classifier:

```text
P1 slots -> attention pooling or max pooling per species -> sigmoid head
```

We should still compare against encoder-only classifiers to verify that P1 is
actually adding value.

## Why This Is Not Just A Classifier

A standard classifier can learn shortcuts:

- background correlation,
- site or recorder artifacts,
- common species priors,
- frequency bands that correlate with labels but do not separate sources.

Factorize-and-Compose Audio-JEPA adds structural pressure:

1. P1 must expose separate latent components.
2. P2 must recombine those components into a mixture latent.
3. Component targets come from individual species recordings.
4. Soundscape JEPA keeps the representation grounded in the target domain.
5. The classifier reads from factored slots instead of an entangled pooled
   vector.

The model is therefore encouraged to learn a representation that is useful for
both explanation and prediction.

## Minimal Experiment Plan

Start simple and add constraints only when the baseline is stable.

1. Train a scratch log-mel CNN or small patch transformer with BCE on
   multi-label soundscape windows.
2. Add synthetic mixtures from train audio and background soundscapes.
3. Add the EMA target encoder.
4. Train P1 slots with factored matching against known synthetic components.
5. Add P2 and the composed latent loss.
6. Add masked soundscape JEPA.
7. Add natural soundscape labels and the final multi-label head.
8. Compare downstream performance using encoder pooled features, P1 slots, P2
   latent, and encoder plus P1.

The first success criterion is not leaderboard performance. The first success
criterion is whether P1 slots become meaningfully class-aligned and improve
multi-label classification over the encoder-only baseline.

## Open Questions

- How many slots should P1 produce?
- Should slots be species-specific, class-agnostic, or a mixture of both?
- Should background get explicit slots?
- Should component targets be individual clips, species prototypes, or
  multiple prototypes per species?
- Should composition be a simple normalized sum or a learned function?
- How strong should the additivity loss be, given that real soundscapes are not
  perfectly additive in log-mel space?
- Does masked JEPA help once composition training exists, or does it compete
  with the factorization objective?
- Is the best downstream representation P1, P2, E_c, or a combination?
- How do we prevent rare species from being absorbed into background slots?

## Working Name

Use **Factorize-and-Compose Audio-JEPA** as the working name.

Short description:

> A scratch-trained audio JEPA for BirdCLEF-style soundscapes where the
> predictor first produces factored species/event latents, then composes them
> into a mixture latent that is trained against additive targets built from
> individual recordings and soundscape context.
