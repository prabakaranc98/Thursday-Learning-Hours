# Resources

## Data

- BirdCLEF 2026 on ImageCLEF:
  <https://www.imageclef.org/BirdCLEF2026>
- BirdCLEF 2026 on Kaggle:
  <https://www.kaggle.com/competitions/birdclef-2026/overview>

Keep downloaded competition data outside git under a local `data/` directory.
The root `.gitignore` excludes common dataset and checkpoint artifacts.

## Python Stack

- `torch` and `torchaudio` for modeling and audio transforms.
- `librosa` for quick spectrogram inspection and baseline preprocessing.
- `numpy`, `pandas`, and `scikit-learn` for data handling and evaluation.
- `tqdm` for experiment progress.

## Modeling Choices

Start simple:

- Audio window length: 5 seconds.
- Sample rate: 32 kHz if matching BirdCLEF-style pipelines, otherwise keep the
  source rate fixed for the first demo.
- Input representation: log-mel spectrogram.
- Backbone: small CNN or patch transformer.
- SSL objective: predict masked target embeddings from visible context
  embeddings.
- Evaluation: linear probe, shallow classifier, or lightweight fine-tune.

## Expected Outputs

- A small pretraining loss curve.
- A baseline supervised classifier result.
- A JEPA-pretrained classifier result.
- A short discussion of what did and did not transfer.
