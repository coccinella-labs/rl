from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub import HfApi
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
    language="en",
    license="mit",
    library_name="custom",
    tags=["reinforcement-learning", "cartpole", "cma-es", "gymnasium"],
    datasets=["gymnasium/CartPole-v1"],
    model_name="CartPole-v1 CMA-ES Solution",
    developers="Niladri Das",
    model_type="Linear Policy",
    repo="https://huggingface.co/harpertoken/pole",
)

# Create and populate the model card
card = ModelCard("# CartPole-v1 CMA-ES Solution\n\n")

# Model Summary with Training Convergence Graph
card.text += """
This model provides a solution to the CartPole-v1 environment using CMA-ES
(Covariance Matrix Adaptation Evolution Strategy), achieving perfect performance
with a simple linear policy. The implementation demonstrates how evolutionary
strategies can effectively solve classic control problems with minimal
architecture complexity.

### Training Convergence
![Training Convergence](assets/training_convergence.png)
*Figure: Training convergence showing the mean fitness (episode length) across
generations. The model achieves optimal performance (500 steps) within 3
generations.*

## Model Details

### Model Description

This is a linear policy model for the CartPole-v1 environment that:
- Uses a simple weight matrix to map 4D state inputs to 2D action outputs
- Achieves optimal performance (500/500 steps) consistently
- Was optimized using CMA-ES, requiring only 3 generations for convergence
- Demonstrates sample-efficient learning for the CartPole balancing task

- **Developed by:** Niladri Das
- **Model type:** Linear Policy
- **Language:** Python
- **License:** MIT
- **Finetuned from model:** No (trained from scratch)

### Model Sources

- **Repository:** https://huggingface.co/harpertoken/pole

## Uses

### Direct Use

The model is designed for:
1. Solving the CartPole-v1 environment from Gymnasium
2. Demonstrating CMA-ES optimization for RL tasks
3. Serving as a baseline for comparison with other algorithms
4. Educational purposes in evolutionary strategies

### Out-of-Scope Use

The model should not be used for:
1. Complex control tasks beyond CartPole
2. Real-world robotics applications
3. Tasks requiring non-linear policies
4. Environments with partial observability

## Bias, Risks, and Limitations

### Technical Limitations
- Limited to CartPole-v1 environment
- Requires full state observation
- Linear policy architecture
- No transfer learning capability
- Environment-specific solution

### Performance Limitations
- May not handle significant environment variations
- No adaptation to changing dynamics
- Limited by linear policy capacity
- Requires precise state information

### Recommendations

Users should:
1. Only use for CartPole-v1 environment
2. Ensure full state observability
3. Understand the limitations of linear policies
4. Consider more complex architectures for other tasks
5. Validate performance in their specific setup

## How to Get Started with the Model

```python
import numpy as np
from gymnasium import make

# Load model weights
weights = np.load('model_weights.npy')  # 4x2 matrix

# Create environment
env = make('CartPole-v1')

# Run inference
def get_action(observation):
    logits = observation @ weights
    return int(logits[0] < logits[1])

observation, _ = env.reset()
while True:
    action = get_action(observation)
    observation, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
```

## Training Details

### Training Data

- **Environment:** Gymnasium CartPole-v1
- **State Space:** 4D continuous (cart position, velocity, pole angle, angular velocity)
- **Action Space:** 2D discrete (left, right)
- **Reward:** +1 for each step, max 500 steps
- **Episode Termination:** Pole angle > 15°, cart position > 2.4, or 500 steps reached

### Training Procedure

#### Training Hyperparameters

- **Algorithm:** CMA-ES
- **Population size:** 16
- **Number of generations:** 15 (early convergence achieved)
- **Initial step size:** 0.5
- **Parameters:** 8 (4x2 weight matrix)
- **Training regime:** Single precision (fp32)

#### Hardware Requirements

- **CPU:** Single core sufficient
- **Memory:** <100MB RAM
- **GPU:** Not required
- **Training time:** ~5 minutes on standard CPU

### Evaluation

#### Testing Data & Metrics

- **Environment:** Same as training (CartPole-v1)
- **Episodes:** 100 test episodes
- **Metrics:** Episode length, success rate

#### Results

- **Average Episode Length:** 500.0 ±0.0
- **Success Rate:** 100%
- **Convergence:** Achieved in 3 generations
- **Final Population Mean:** 484.64
- **Best Performance:** 500/500 consistently

## Environmental Impact

- **Training time:** ~5 minutes
- **Hardware:** Standard CPU
- **Energy consumption:** Negligible (<0.001 kWh)
- **CO2 emissions:** Minimal (<0.001 kg)

## Citation

**BibTeX:**
```bibtex
@misc{das2024cartpole,
  author = {Niladri Das},
  title = {CartPole-v1 CMA-ES Solution},
  year = {2024},
  publisher = {Hugging Face},
  journal = {Hugging Face Model Hub},
  howpublished = {https://huggingface.co/harpertoken/pole}
}
```
"""

# Push to hub
card.push_to_hub(REPO_ID)
print(f"Successfully pushed model card to {REPO_ID}")
