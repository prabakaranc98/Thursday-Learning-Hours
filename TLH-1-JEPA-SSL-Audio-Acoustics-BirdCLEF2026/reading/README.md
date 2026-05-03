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

## Reading Lens

While reading, track these questions:

- What is predicted: waveform samples, spectrogram bins, tokens, or latent
  embeddings?
- What makes the prediction non-trivial?
- How does the method avoid collapse?
- What augmentations or masking policies are domain-specific to audio?
- How would the representation be evaluated for species identification?
