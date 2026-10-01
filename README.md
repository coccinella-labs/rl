<p align="center">
  <img src="https://raw.githubusercontent.com/coccinella-labs/rl/main/.github/assets/thumbnail.png" alt="rl" width="100%">
</p>

# Harpertoken RL

CMA-ES training for CartPole-v1 with a linear policy, plus an evaluator and a small
FastAPI service. The package is published as `harpertoken-pole` 0.1.0 and supports Python
3.8 and newer.

## Read this before loading a published model

The two loaders in this repository disagree about the file layout, and neither matches
[`harpertoken/pole`](https://huggingface.co/harpertoken/pole) as published.

| Loader | Expects | On `harpertoken/pole` |
|---|---|---|
| `CMAESAgent.from_pretrained` in `harpertoken/models/model.py` | `weights.npy` + `metadata.json` | both absent |
| `_from_pretrained` in `harpertoken/evaluation/test_model.py` | `model.npy` | present |

The Hub repository ships `model.npy`, `cmaes_model.npy`, `cmaes_model.pth`, a
convergence figure and a video. It has no `weights.npy` and no `metadata.json`, so
`CMAESAgent.from_pretrained("harpertoken/pole")` cannot work. It actually fails twice over:
first with a `TypeError` because `_from_pretrained` declares `proxies` and
`resume_download` as required keyword arguments that the mixin never passes, and then,
once those are supplied by hand, with a 404 for the missing `metadata.json`.

The missing piece is `metadata.json`. Its contents are fully determined by the
environment, four fields:

```json
{"observation_space": 4, "action_type": "discrete", "env_name": "CartPole-v1", "num_actions": 2}
```

To load from the Hub, add `weights.npy` (a flat copy of `model.npy`) and that
`metadata.json` to the `pole` repository, then use the models loader. Until then, load
the array yourself and assign it:

```python
import numpy as np
from harpertoken.models.model import CMAESAgent

agent = CMAESAgent("CartPole-v1")
agent.weights = np.load(
    hf_hub_download("harpertoken/pole", "model.npy"), allow_pickle=True
)
agent.observation_space = 4
agent.num_actions = 2
agent.action_space = 2

mean, std = agent.evaluate(num_episodes=100)
```

`model.npy` is stored flat as eight float64 values. `CMAESAgent.get_action` reshapes it
to `(4, 2)`, so pass it through unchanged rather than reshaping it yourself.

## The weights are not interchangeable

Three files in `harpertoken/pole` describe two different solutions. `model.npy` is the
one that scores 500. `cmaes_model.npy` and `cmaes_model.pth` hold the CMA-ES search state,
including a recorded fitness of 500.0, but they describe a weight vector that averages
498.99 with a standard deviation of 5.93 over 100 episodes, dipping to 451. The recorded
fitness is optimistic relative to what those weights actually do. Use `model.npy`.

Reading the `.npy` and `.pth` files needs `allow_pickle=True`. `scripts/inspect_model.py`
prints all three so you can compare them without guessing.

## Layout

| Path | Contents |
|---|---|
| `harpertoken/models/model.py` | `CMAESAgent`, the `ModelHubMixin` implementation |
| `harpertoken/training/train.py` | CMA-ES loop over generations and candidates |
| `harpertoken/training/train_model.py` | Training entry point, writes `metadata.json` |
| `harpertoken/evaluation/test_model.py` | Second agent implementation using `model.npy` |
| `harpertoken/hub/push_to_hub.py` | Publishes a checkpoint to the Hub |
| `harpertoken/hub/model_card.py` | Generates the model card |
| `scripts/inspect_model.py` | Prints the three weight files side by side |
| `deploy/` | Dockerfile, `app.py`, Kubernetes manifest |
| `tests/` | Pytest suite |

## Training

There are two training entry points, and they are not the same code.

```sh
pip install -r requirements.txt
python -m harpertoken.training.main     # 50 iterations, saves cmaes_model.npy
python harpertoken/training/train.py    # 100 generations, saves cartpole_cmaes/
```

`training/main.py` defines its own `CMAESAgent` class that talks to
`CMAEvolutionStrategy` directly, runs 50 iterations, and writes the best parameters to
`cmaes_model.npy`. That is the file whose recorded fitness of 500.0 is optimistic, per
the section above.

`training/train.py` uses the shared `CMAESAgent` from `models/model.py`, runs 100
generations with a population of 16, and scores each candidate on 5 episodes. It writes
`weights.npy` and `metadata.json` to `cartpole_cmaes/`, which is the layout
`from_pretrained` expects. CMA-ES minimises the negated reward, so the optimiser descends
toward 500.

## Evaluation

```python
from harpertoken.models.model import CMAESAgent

agent = CMAESAgent("CartPole-v1")
agent.weights = ...  # see the loading section above
mean, std = agent.evaluate(num_episodes=100)
```

## Serving

`deploy/app.py` exposes the policy over HTTP, and `deploy/k8s_deployment.yaml` runs it on
a cluster. Build the image with the `Dockerfile` in `deploy/`.

## Tests and CI

```sh
pip install pytest pytest-asyncio
pytest tests/
```

`.github/workflows/ci.yml` runs the suite on Python 3.8 through 3.12. pre-commit covers
formatting. A separate workflow builds the Docker image.

## License

MIT. See [LICENSE](LICENSE).