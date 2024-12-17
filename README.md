# Be More Diverse Than The Most Diverse: Online Selection of Diverse Mixtures of Generative Models
This repository contains the codebase for the paper ????. The script demonstrates how to find the optimal mixture in an offline and online manner to choose the optimal mixture of several generative models.

## Abstract
The availability of multiple training algorithms and architectures for generative models requires a selection mechanism to form a single model over a group of well-trained generation models. The selection task is commonly addressed by identifying the model that maximizes an evaluation score based on the diversity and quality of the generated data. However, such a best-model identification approach overlooks the possibility that a mixture of available models can outperform each individual model. In this work, we explore the selection of a mixture of multiple generative models and formulate a quadratic optimization problem to find an optimal mixture model achieving the maximum of kernel-based evaluation scores including kernel inception distance (KID) and Renyi kernel entropy (RKE). To identify the optimal mixture of the models using the fewest possible sample queries, we propose an online learning approach called **Mixture Upper Confidence Bound (Mixture-UCB)**. Specifically, our proposed online learning method can be extended to every convex quadratic function of the mixture weights, for which we prove a concentration bound to enable the application of the UCB approach. We prove a regret bound for the proposed Mixture-UCB algorithm and perform several numerical experiments to show the success of the proposed Mixture-UCB method in finding the optimal mixture of text-based and image-based generative models.  

## Requirements

- Python 3.x
- numpy
- torch
- cvxpy
- sklearn

## Offline Mixture
Example code as follows:
```python
import numpy as np
from offline_mixture import calculate_optimal_mixture, calculate_rke, calculate_precision

# Generate random normal distribution as generated and real data
models = {
        "model_0": np.random.normal(loc=0.1, scale=1, size=(200,5)),
        "model_1": np.random.normal(loc=-0.5, scale=1, size=(200,5)),
        "model_2": np.random.normal(loc=100, scale=0.5, size=(200,5))
    }
real_data = np.random.normal(loc=0, scale=0.3, size=(200, 5))

# Calculate optimal mixture
optimal_alphas = calculate_optimal_mixture(
    models,
    quadratic_calculator=calculate_rke,
    has_linear=False,
    real_data=real_data,
    linear_term_calculator=calculate_precision,
    linear_term_weight=0.05,
    sigma=10
)

print("Optimal Mixture Coefficients:", optimal_alphas)
```
