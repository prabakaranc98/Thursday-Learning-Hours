# Artifacts

Architecture and method diagrams for Factorize-and-Compose Audio-JEPA.

## Diagrams

| File | Description |
|---|---|
| `architecture_overview` | Full model: input pipeline -> E_c -> P1 slots -> P2 + Classifier, with EMA target encoder path |
| `mode2_composition` | Mode 2 (core new idea): composition from known components, triangle consistency |
| `geometry` | Representation geometry: unit sphere, positive cone, triangle consistency constraint |

## Compile

```bash
cd artifacts
bash compile.sh
```

Requires TeX Live with TikZ. Outputs PDFs to `pdf/`, PNGs to `png/`.

## Color Legend

| Color | Component |
|---|---|
| Teal | Audio input nodes |
| Blue | Context encoder E_c |
| Gray | Target encoder E_t (EMA / stop-grad) |
| Orange-Red | P1 slot attention |
| Purple | P2 composition predictor |
| Green | Classifier head |
| Light orange | Slot nodes |
| Dark red | Loss labels |
