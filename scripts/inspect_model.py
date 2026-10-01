"""Inspect the weight artifacts published to the harpertoken/pole Hub repo.

The repository ships three files:

  model.npy         the policy that reaches the 500-step ceiling, stored flat
                    as 8 float64 values; the matrix form is reshape(4, 2)
  cmaes_model.npy   the CMA-ES state (weights, recorded fitness, intended
                    shape) as a pickled object, so it needs allow_pickle
  cmaes_model.pth   the same state as a torch archive

There is no metadata.json or weights.npy in the repository.
"""

import numpy as np

REPO = "harpertoken/pole"


def main() -> None:
    from huggingface_hub import hf_hub_download

    policy_path = hf_hub_download(REPO, "model.npy")
    weights = np.load(policy_path)
    matrix = weights.reshape(4, 2)

    print(f"{REPO}/model.npy")
    print(f"  shape      {weights.shape} {weights.dtype} (flat)")
    print(f"  matrix     {matrix.shape}")
    print(f"  range      [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"  values     {weights}")

    state_path = hf_hub_download(REPO, "cmaes_model.npy")
    state = np.load(state_path, allow_pickle=True).item()

    print(f"\n{REPO}/cmaes_model.npy")
    print(f"  keys       {sorted(state)}")
    print(f"  fitness    {state['fitness']} (recorded during the search)")
    print(f"  shape      {tuple(state['shape'])}")
    print(f"  values     {np.asarray(state['weights'])}")


if __name__ == "__main__":
    main()
