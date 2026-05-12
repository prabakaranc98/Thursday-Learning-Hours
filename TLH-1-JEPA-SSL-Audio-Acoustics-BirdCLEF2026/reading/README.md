# Reading

## Required

1. I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding
   Predictive Architecture  
   <https://arxiv.org/abs/2301.08243>
2. AudioMAE: Masked Autoencoders that Listen  
   <https://arxiv.org/abs/2207.06405>
3. Audio Spectrogram Transformer  
   <https://arxiv.org/abs/2104.01778>

## Useful Background

1. wav2vec 2.0  
   <https://arxiv.org/abs/2006.11477>
2. HuBERT  
   <https://arxiv.org/abs/2106.07447>
3. BYOL-A  
   <https://arxiv.org/abs/2103.06695>

## Applied Context

1. BirdCLEF++ / BirdCLEF 2026 task page  
   <https://www.imageclef.org/BirdCLEF2026>
2. BirdCLEF 2026 Kaggle page  
   <https://www.kaggle.com/competitions/birdclef-2026/overview>

## Use-Case References

These references are most relevant to the BirdCLEF soundscape setting: long
passive recordings, short prediction windows, weak labels, domain shift between
curated species clips and field soundscapes, and multi-label prediction.

1. BirdCLEF 2021 main task: focal recordings to five-second soundscape
   prediction, with explicit discussion of the domain mismatch between short
   species recordings and long soundscapes.  
   <https://www.imageclef.org/BirdCLEF2021MainTask>
2. Overview of BirdCLEF 2024: acoustic identification in Western Ghats
   soundscapes; useful for domain adaptation, limited labeled data,
   pseudo-labeling, and competition constraints.  
   <https://ceur-ws.org/Vol-3740/paper-184.pdf>
3. Overview of BirdCLEF+ 2025: multi-taxonomic sound identification; useful
   because the task moves beyond birds to overlapping insects, amphibians,
   mammals, and birds in passive soundscapes.  
   <https://ceur-ws.org/Vol-4038/paper_232.pdf>
4. Methods for Training Convolutional Neural Networks to Identify Bird Species
   in Complex Soundscape Recordings: practical CNN training choices,
   augmentation, mixing, weighting, and weak-label handling.  
   <https://ceur-ws.org/Vol-3740/paper-193.pdf>
5. Improving Bird Recognition using Pseudo-Labeled Recordings from the Target
   Location: target-domain pseudo-labeling for soundscape domain adaptation.  
   <https://ceur-ws.org/Vol-3740/paper-199.pdf>
6. Addressing the Challenges of Domain Shift in Bird Call Classification for
   BirdCLEF 2024: analysis of why curated training audio differs from passive
   acoustic monitoring test soundscapes.  
   <https://ceur-ws.org/Vol-3740/paper-211.pdf>
7. TUC Media Computing at BirdCLEF 2024: Improving Birdsong Classification
   Through Single Learning Models: useful as a single-model reference rather
   than an ensemble/pretrained-heavy system.  
   <https://ceur-ws.org/Vol-3740/paper-206.pdf>

## Scratch-First Architecture References

For this TLH, treat these as algorithmic and architectural references. We can
use the encoder shapes, objectives, masking policies, and evaluation ideas
without loading pretrained checkpoints.

1. Audio Spectrogram Transformer: a spectrogram patch transformer architecture
   that can be initialized from scratch for our experiment.  
   <https://arxiv.org/abs/2104.01778>
2. AudioMAE: masked autoencoding over audio spectrogram patches; useful as a
   reconstruction-based SSL baseline.  
   <https://arxiv.org/abs/2207.06405>
3. A-JEPA: applies JEPA-style latent prediction to audio spectrogram patches
   with context and target encoders.  
   <https://arxiv.org/abs/2311.15830>
4. Audio-JEPA: a later audio-specific JEPA reference using masked mel
   spectrogram patch prediction in latent space.  
   <https://arxiv.org/abs/2507.02915>

## Pretrained Baselines To Compare Against Later

These are not part of the first scratch-first build. They are useful as strong
reference baselines after our own architecture is implemented.

1. Perch 2.0: a supervised bioacoustics foundation model with off-the-shelf
   classification scores and transfer-learning embeddings across many
   vocalizing species.  
   <https://research.google/pubs/perch-20-the-bittern-lesson-for-bioacoustics/>
2. Google Research Perch repository: implementation notes, PCEN frontend,
   EfficientNet model family, Kaggle model links, and agile-modeling tooling.  
   <https://github.com/google-research/perch>
3. Global birdsong embeddings enable superior transfer learning for
   bioacoustic classification: original Perch transfer-learning reference.  
   <https://www.nature.com/articles/s41598-023-49989-z>

## Constraint For Our Build

Do not start by importing a pretrained BirdNET, Perch, EnCodec, AudioMAE,
AudioSet, or ImageNet checkpoint. The first implementation should train its
own audio encoder from the provided audio using the algorithm we define:

1. supervised baseline from log-mel windows,
2. SSL or JEPA pretraining on soundscape windows,
3. supervised fine-tuning with species-specific train audio,
4. optional prototype, eventness, and pseudo-labeling auxiliary losses.

See [`../factor-compose-audio-jepa.md`](../factor-compose-audio-jepa.md) for
the working architecture note that turns this constraint into a factored,
compositional JEPA proposal.

## Reading Lens

While reading, track these questions:

- What is predicted: waveform samples, spectrogram bins, tokens, or latent
  embeddings?
- What makes the prediction non-trivial?
- How does the method avoid collapse?
- What augmentations or masking policies are domain-specific to audio?
- How would the representation be evaluated for species identification?
- Would the method learn better from long target-domain soundscapes, curated
  species clips, or both?
- How would the learned representation support multi-label prediction over
  soundscape windows?
