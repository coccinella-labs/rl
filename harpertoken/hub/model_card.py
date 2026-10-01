from huggingface_hub import HfApi, ModelCard, ModelCardData
import os

# Initialize Hugging Face API
api = HfApi()
REPO_ID = "harpertoken/pole"

# First, upload the convergence plot
plot_path = "plots/training_convergence.png"
if os.path.exists(plot_path):
    api.upload_file(
        path_or_fileobj=plot_path,
        path_in_repo="assets/training_convergence.png",
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"Successfully uploaded {plot_path} to {REPO_ID}")
else:
    print(f"Warning: {plot_path} not found")

# Create model card with metadata
card_data = ModelCardData(
    language='en',
    license='mit',
    library_name='custom',
    pipeline_tag='reinforcement-learning',
    datasets=['gymnasium/CartPole-v1'],
    tags=['cma-es', 'cartpole', 'evolutionary-strategy', 'gymnasium'],
    repo='https://huggingface.co/harpertoken/pole',
)

# Create and populate the model card
card = ModelCard("""\
# pole

A linear policy for `CartPole-v1`, found by covariance matrix adaptation evolution strategy. The policy is a single 4-by-2 matrix of weights mapping the four-dimensional observation — cart position, cart velocity, pole angle, pole angular velocity — to a score for each of the two actions, with the larger score taken. Eight numbers in total.

Evaluating `model.npy` over 100 episodes, resetting with seeds 0 through 99, the mean episode length is 500.00 with a standard deviation of 0.00; all 100 episodes reach the environment's 500-step ceiling. Since 500 is the maximum reward `CartPole-v1` permits, this is the ceiling rather than a measured score against a baseline, and there is nothing further to optimise.

The repository contains three representations of a solution and they are not identical. `model.npy` is the one that achieves the result above and is the one to use. It is stored flat, as eight float64 values; the matrix form is `reshape(4, 2)`, which the previous version of this card did not say and which its example code got wrong by treating the file as a matrix directly. `cmaes_model.npy` and `cmaes_model.pth` hold the CMA-ES state — weights, a recorded fitness of 500.0, and the intended shape — but they describe a different weight vector. Measured over the same 100 episodes that vector averages 498.99 with a standard deviation of 5.93, touching 500 in 96 episodes and falling as low as 451. The recorded fitness of 500.0 is therefore slightly optimistic relative to what these weights actually do, and the two files should not be treated as interchangeable. Both `.npy` and `.pth` require `allow_pickle=True` / a permissive load to read.

No Python code is included. The earlier card showed a `CMAESAgent` class and imported it `from model`, and referenced a `model_weights.npy`; neither the module nor that filename has existed in this repository. What is here is the weight arrays, a convergence figure under `assets/`, and a short video of the policy running.

## Usage

```python
import numpy as np, gymnasium as gym

weights = np.load("model.npy").reshape(4, 2)   # stored flat as 8 float64 values

env = gym.make("CartPole-v1")
obs, _ = env.reset(seed=0)
steps = 0
while True:
    obs, reward, terminated, truncated, info = env.step(int(np.argmax(obs @ weights)))
    steps += 1
    if terminated or truncated:
        break
print(steps)
```

## Limitations

This solves one environment, from full state observation, with a policy class of eight parameters. It does not transfer: change the dynamics, the observation space or the action space and the weights are meaningless, and there is no mechanism to relearn them. It says nothing in particular about linear policies in control — the earlier card claimed this result demonstrated that CartPole's optimal policy is approximately linear, which is a much broader claim than one successful parameter vector supports, and has been removed. `assets/training_convergence.png` plots mean fitness per generation; the search reached the ceiling within roughly three generations from a population of sixteen, which is unremarkable for a problem of this size and not evidence about CMA-ES in general.

## Attribution

CartPole-v1 is the classic control task from Barto, Sutton and Anderson, and the Gymnasium implementation is described in Towers et al., *Gymnasium: A Standard Interface for Reinforcement Learning Environments* (2024). CMA-ES follows Hansen, *The CMA Evolution Strategy: A Tutorial* (2016).
""")

# Push to hub
card.push_to_hub(REPO_ID)
print(f"Successfully pushed model card to {REPO_ID}")
