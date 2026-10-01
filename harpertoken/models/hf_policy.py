"""Transformers-compatible serialisation of the CartPole linear policy.

The CMA-ES policy learned for CartPole-v1 is a single linear map from the
four observation features to two action logits, so it is stored here as a
one-layer `torch.nn.Linear` rather than a stack of transformer blocks.
Attending over four scalars has nothing to relate, so an actual transformer
would add parameters and training steps without changing the answer. The
card states that the policy is linear.

This class exists so the weights can be read through the standard
`from_pretrained` machinery and stored as `config.json` plus
`model.safetensors`. Registering it with `AutoModel` happens on import of
this module; see `register()`.
"""

import os

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)

CONFIG_NAME = "config.json"


class CmaesLinearPolicyConfig(PretrainedConfig):
    """Configuration for the CartPole linear policy."""

    model_type = "cmaes_linear_policy"

    def __init__(
        self,
        observation_space=4,
        num_actions=2,
        env_name="CartPole-v1",
        action_type="discrete",
        **kwargs,
    ):
        self.observation_space = observation_space
        self.num_actions = num_actions
        self.env_name = env_name
        self.action_type = action_type
        super().__init__(**kwargs)


class CmaesLinearPolicy(PreTrainedModel):
    """Linear action-scoring policy, equivalent to the CMA-ES solution.

    The forward pass reproduces `CMAESAgent.get_action` for a discrete action
    space: logits are the observation dotted with the weight matrix, and the
    chosen action is the argmax. `get_action` reshapes the stored eight-element
    vector to (observation_space, num_actions); here that matrix is the
    transposed `nn.Linear` weight, which is why the saved tensor has shape
    (num_actions, observation_space).
    """

    config_class = CmaesLinearPolicyConfig
    base_model_prefix = "policy"

    # Transformers 5 reads this mapping while finalising a load. Nothing here is
    # tied, but the attribute has to exist or `from_pretrained` raises
    # AttributeError before the weights are placed.
    all_tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)
        self.policy = nn.Linear(config.observation_space, config.num_actions)

    def forward(self, observation, **kwargs):
        """Return action logits for one observation or a batch of them."""
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        return self.policy(observation)

    def get_action(self, observation):
        """Single-observation convenience wrapper returning an int."""
        with torch.inference_mode():
            logits = self.forward(torch.as_tensor(observation, dtype=torch.float32))
        return int(torch.argmax(logits[0]))

    @classmethod
    def from_flat_weights(cls, weights, **kwargs):
        """Build a policy from the flat eight-element CMA-ES vector."""
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        config = kwargs.pop(
            "config",
            CmaesLinearPolicyConfig(
                observation_space=int(weights.size // 2),
                num_actions=2,
                **kwargs,
            ),
        )
        model = cls(config)
        matrix = weights.reshape(config.observation_space, config.num_actions)
        with torch.no_grad():
            model.policy.weight.copy_(torch.as_tensor(matrix.T, dtype=torch.float32))
            model.policy.bias.zero_()
        model.eval()
        return model

    def to_flat_weights(self):
        """Return the eight-element vector `CMAESAgent` expects."""
        with torch.inference_mode():
            matrix = self.policy.weight.detach().cpu().numpy().T
        return matrix.reshape(-1).astype(np.float64)


def register():
    """Register this architecture so `AutoModel`/`AutoConfig` resolve it.

    Both registrations are required: without the config, `AutoConfig` cannot
    map the `model_type` in `config.json` back to a class, and `AutoModel`
    then has nothing to look up.
    """
    AutoConfig.register(
        CmaesLinearPolicyConfig.model_type,
        CmaesLinearPolicyConfig,
        exist_ok=True,
    )
    AutoModel.register(
        CmaesLinearPolicyConfig,
        CmaesLinearPolicy,
        exist_ok=True,
    )


def save_weights(model, output_dir):
    """Write `config.json` and `model.safetensors` into `output_dir`."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    return os.path.join(output_dir, "model.safetensors")


register()
