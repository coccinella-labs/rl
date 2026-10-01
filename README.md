<p align="center">
  <img src="https://raw.githubusercontent.com/Coccinella-Labs/rl/main/.github/assets/thumbnail.png" alt="rl" width="100%">
</p>

Solves CartPole-v1 with CMA-ES and a linear policy (`harpertoken-pole` 0.1.0, Python 3.8+). The trained weights are published to [`harpertoken/pole`](https://huggingface.co/harpertoken/pole) as `model.npy`, stored flat as eight float64 values, alongside the CMA-ES search state in `cmaes_model.npy` and `cmaes_model.pth`. There is no `metadata.json`.

## Layout

- `harpertoken/` - training (`training/`), evaluation (`evaluation/`), models (`models/`), data (`data/`), hub (`hub/`)
- `tests/`, `scripts/` - tests and helpers
- `deploy/` - deployment assets
